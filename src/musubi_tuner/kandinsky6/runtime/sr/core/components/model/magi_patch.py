"""Runtime patch for MagiCompiler static-shape marking (shared by the magi builders)."""

def patch_magi_mark_static_shapes(*, mark_owner: bool = False):
    """
    Runtime patch for magi_compiler._api._mark_static_shapes.

    It avoids recursively walking the whole model / pipeline object graph,
    which can contain urllib3, tokenizers, FSDP internals, hooks, caches, etc.

    mark_owner=False is intentional and recommended.
    """
    print(f"USING MAGI PATCH")
    import torch
    import magi_compiler._api as magi_api

    # Avoid applying the patch multiple times.
    current_fn = magi_api._mark_static_shapes
    if getattr(current_fn, "_patched_skip_unsafe_traversal", False):
        return

    original_fn = current_fn

    def patched_mark_static_shapes(bound, dynamic_records, owner=None):
        """
        Mark static dimensions for tensors that are not marked as dynamic.

        dynamic_records maps:
            id(tensor) -> set(dynamic_dim_indices)
        """

        visited = set()

        scalar_types = (
            int,
            float,
            complex,
            str,
            bytes,
            bool,
            type(None),
        )

        def mark_tensor_static(tensor):
            tensor_id = id(tensor)
            dyn_dims = dynamic_records.get(tensor_id, set())

            for dim_idx in range(tensor.ndim):
                if dim_idx not in dyn_dims:
                    torch._dynamo.mark_static(tensor, dim_idx)

        def traverse_and_mark(obj):
            obj_id = id(obj)

            if obj_id in visited:
                return

            if isinstance(obj, scalar_types):
                return

            visited.add(obj_id)

            if isinstance(obj, torch.Tensor):
                mark_tensor_static(obj)
                return

            # Important:
            # Do not traverse arbitrary torch.nn.Module.__dict__.
            # It may contain FSDP internals, tokenizers, HTTP clients,
            # urllib3 objects, hooks, caches, etc.
            if isinstance(obj, torch.nn.Module):
                containers = (
                    getattr(obj, "_parameters", None),
                    getattr(obj, "_buffers", None),
                    getattr(obj, "_modules", None),
                )

                for container in containers:
                    if isinstance(container, dict):
                        try:
                            values = list(container.values())
                        except Exception:
                            continue

                        for val in values:
                            traverse_and_mark(val)

                return

            # Only handle real dicts / OrderedDict-like dict subclasses.
            # Do NOT use collections.abc.Mapping here.
            # urllib3 has mapping-like objects that raise:
            # NotImplementedError: Iteration over this class is unlikely to be threadsafe.
            if isinstance(obj, dict):
                try:
                    values = list(obj.values())
                except Exception:
                    return

                for val in values:
                    traverse_and_mark(val)

                return

            if isinstance(obj, (list, tuple, set, frozenset)):
                try:
                    items = tuple(obj)
                except Exception:
                    return

                for item in items:
                    traverse_and_mark(item)

                return

            # Defensively traverse simple custom objects / dataclasses.
            # This is useful for objects like sparse_params.
            try:
                obj_dict = vars(obj)
            except Exception:
                obj_dict = None

            if isinstance(obj_dict, dict):
                try:
                    values = list(obj_dict.values())
                except Exception:
                    return

                for val in values:
                    traverse_and_mark(val)

                return

            # Defensively traverse __slots__.
            slots = getattr(type(obj), "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)

            try:
                slots = tuple(slots)
            except Exception:
                return

            for slot in slots:
                if slot in ("__dict__", "__weakref__"):
                    continue

                try:
                    val = getattr(obj, slot)
                except Exception:
                    continue

                traverse_and_mark(val)

        # Mark tensors from real call arguments.
        for arg_val in bound.arguments.values():
            traverse_and_mark(arg_val)

        # Recommended: skip owner traversal.
        #
        # Your errors happen because Magi recursively walks `owner`,
        # i.e. the module / pipeline object graph, and reaches unrelated objects
        # such as urllib3 containers.
        if mark_owner and owner is not None:
            traverse_and_mark(owner)

    patched_mark_static_shapes._patched_skip_unsafe_traversal = True
    patched_mark_static_shapes._original = original_fn

    magi_api._mark_static_shapes = patched_mark_static_shapes
