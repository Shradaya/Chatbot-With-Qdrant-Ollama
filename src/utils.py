def remove_stop_words(text: str) -> list:
        return list(set(text.lower()\
                .replace(" to ", " ")\
                .replace(" as ", " ")\
                .replace(" be ", " ")\
                .replace(" by ", " ")\
                .replace(" in ", " ")\
                .replace(" or ", " ")\
                .replace(" of ", " ")\
                .replace(" with ", " ")\
                .replace(" and ", " ")\
                .replace(" from ", " ")\
                .replace(" the ", " ")\
                .replace(" etc ", " ")\
                .replace(" other ", " ")\
                .split(" ")))