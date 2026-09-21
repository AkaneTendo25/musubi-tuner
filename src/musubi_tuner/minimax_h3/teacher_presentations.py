"""Caption declarations for opt-in privileged H3 teacher presentations."""


def ref_teacher_caption(caption: str, *, has_audio: bool = True) -> str:
    audio_definition = " <Audio 1> is its synchronized soundtrack." if has_audio else ""
    audio_retention = "<Audio 1>: fully_copy - reuse the complete source soundtrack in the target.\n" if has_audio else ""
    summary = "Reproduce <Video 1> and its sound without changes." if has_audio else "Reproduce <Video 1> without changes."
    return (
        "subject_definitions:\n"
        f"<Video 1> is the source of the target video.{audio_definition}\n\n"
        f"summary:\n[video editing] {summary}\n\n"
        "retention_analysis:\n"
        "<Video 1>: fully_preserved - keep every frame, subject, action and camera move.\n"
        f"{audio_retention}\n"
        f"detailed_description:\n{caption}"
    )


def subject_reference_caption(caption: str, image_count: int, *, still_image: bool) -> str:
    if image_count < 1:
        raise ValueError("subject_ref teacher requires at least one reference image")
    labels = [f"<Subject {index}>" for index in range(1, image_count + 1)]
    subjects = " and ".join(labels)
    definitions = "\n".join(
        f"<Subject {index}> takes appearance (face and hair) from <Picture {index}>." for index in range(1, image_count + 1)
    )
    retention = "\n".join(
        f"<Subject {index}>: attribute_transfer - use <Picture {index}> for appearance only; "
        "follow the description for pose, clothing, framing and setting."
        for index in range(1, image_count + 1)
    )
    medium = "still image" if still_image else "video"
    return (
        f"subject_definitions:\n{definitions}\n\n"
        f"summary:\n[reference generation] Create a {medium} showing {subjects}.\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{caption}"
    )
