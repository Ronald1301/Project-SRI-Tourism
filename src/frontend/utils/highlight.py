def highlight_text(text, query):
    words = query.split()

    for w in words:
        text = text.replace(
            w,
            f"[{w}]"
        )

    return text