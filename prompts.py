"""Prompts used with the Ollama vision model."""


def tomato_growth_stage_prompt_AG() -> str:
    """Prompt asking the model to classify a tomato plant's growth stage(s) from an image."""
    return (
        "You are an expert agronomist specializing in tomato cultivation. "
        "Look at the provided image of a tomato plant and identify its current growth stage.\n\n"
        "Choose one or, if the plant clearly shows visual cues for two adjacent stages at once, "
        "two of the following growth stages:\n"
        "1. Side shooting\n"
        "2. Flowering\n"
        "3. Fruit development\n"
        "4. Fruit ripening\n"
        "5. Senescence\n\n"
        "Base your judgment on visible cues such as the presence and size of side shoots, "
        "flowers, green (unripe) fruit, coloring (reddening/ripening) fruit, and majority of yellowing, "
        "wilting foliage. Only report two stages if you can point to distinct visual "
        "evidence for each; otherwise report just one.\n\n"
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        '  "growth_stages": [\n'
        "    {\n"
        '      "growth_stage_number": <integer, 1-5>,\n'
        '      "growth_stage_name": <string, one of "Side shooting", "Flowering", '
        '"Fruit development", "Fruit ripening", "Senescence">\n'
        "    }\n"
        "    // one entry, or two entries if two stages are both visible\n"
        "  ],\n"
        '  "reasoning": <string, one or two sentences explaining the visual evidence you used '
        "for each reported stage>\n"
        "}"
    )


def tomato_growth_stage_schema() -> dict:
    """JSON schema for the tomato growth stage response, for use with Ollama's `format` param."""
    stage_names = [
        "Side shooting",
        "Flowering",
        "Fruit development",
        "Fruit ripening",
        "Senescence",
    ]
    stage_item = {
        "type": "object",
        "properties": {
            "growth_stage_number": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "growth_stage_name": {"type": "string", "enum": stage_names},
        },
        "required": ["growth_stage_number", "growth_stage_name"],
    }
    return {
        "type": "object",
        "properties": {
            "growth_stages": {
                "type": "array",
                "items": stage_item,
                "minItems": 1,
                "maxItems": 2,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["growth_stages", "reasoning"],
    }


def tomato_ripeness_count_prompt() -> str:
    """Prompt asking the model to count tomatoes in an image by ripeness stage."""
    return (
        "You are an expert agronomist specializing in tomato cultivation. "
        "Look at the provided image and count every individual tomato fruit visible, "
        "classifying each one into exactly one of the following ripeness stages:\n\n"
        "1. Raw: the fruit is entirely or predominantly green, with little to no red, "
        "orange, or yellow coloring.\n"
        "2. Semi-ripe: the fruit is transitioning color, showing yellow or orangish-red, "
        "or a mix of red and green — it has started to ripen but is not yet fully red.\n"
        "3. Ripe: the fruit is fully or almost fully red, showing no significant green "
        "or yellow remaining.\n\n"
        "Count only fruit that are clearly visible and identifiable as tomatoes; do not "
        "count fruit that are too small, blurry, or occluded to classify confidently.\n\n"
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        '  "raw_count": <integer, number of raw (green) tomatoes>,\n'
        '  "semi_ripe_count": <integer, number of semi-ripe (yellow/orangish-red) tomatoes>,\n'
        '  "ripe_count": <integer, number of ripe (fully red) tomatoes>,\n'
        '  "total_count": <integer, sum of the three counts above>,\n'
        '  "reasoning": <string, one or two sentences explaining how you distinguished the stages>\n'
        "}"
    )


def tomato_ripeness_count_schema() -> dict:
    """JSON schema for the tomato ripeness count response, for use with Ollama's `format` param."""
    return {
        "type": "object",
        "properties": {
            "raw_count": {"type": "integer", "minimum": 0},
            "semi_ripe_count": {"type": "integer", "minimum": 0},
            "ripe_count": {"type": "integer", "minimum": 0},
            "total_count": {"type": "integer", "minimum": 0},
            "reasoning": {"type": "string"},
        },
        "required": [
            "raw_count",
            "semi_ripe_count",
            "ripe_count",
            "total_count",
            "reasoning",
        ],
    }


def note_time_extraction_prompt() -> str:
    """Prompt asking the model to read the Note/Time overlay burned into the image by NoteCam."""
    return (
        "This image has a small text overlay burned into the bottom-left corner by a "
        "camera timestamp app. The overlay has two lines, each starting with a label:\n"
        '  "Time: <date and time>"\n'
        '  "Note: <a short code/label>"\n\n'
        "Read the overlay text exactly as it appears, preserving punctuation, dashes, "
        "and capitalization. Ignore all other text and objects in the image.\n\n"
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        '  "Note": <string, the exact text after "Note:">,\n'
        '  "Time": <string, the exact text after "Time:">\n'
        "}\n\n"
        'If a label is not visible in the image, use an empty string "" for its value.'
    )


def note_time_extraction_schema() -> dict:
    """JSON schema for the Note/Time overlay extraction response, for use with Ollama's `format` param."""
    return {
        "type": "object",
        "properties": {
            "Note": {"type": "string"},
            "Time": {"type": "string"},
        },
        "required": ["Note", "Time"],
    }


def cross_dap_reid_prompt(num_candidates: int) -> str:
    """Prompt asking the model to re-identify a plant across candidate images taken at different DAP.

    Assumes the model is given `num_candidates + 1` images in order: the query image
    first, followed by the candidates. `num_candidates` must match how many candidate
    images are actually attached, since the schema's valid index range depends on it.
    """
    candidate_lines = "\n".join(
        f"  - Candidate {i}: Image {i + 1}" for i in range(1, num_candidates + 1)
    )
    return (
        "You are an expert in plant phenotyping performing cross-time re-identification "
        "of tomato plants.\n\n"
        f"You are given {num_candidates + 1} images, in this order:\n"
        "  - Image 1: the QUERY image, showing one specific tomato plant.\n"
        f"{candidate_lines}\n\n"
        "Each candidate image was photographed at a different DAP (Days After Planting) and may be at different angle"
        "than the query image, and may show a completely different individual plant. The "
        "SAME plant can look different across DAP values, so judge identity only "
        "using the structural features below.\n\n"
        "Structural features to compare (DAP-invariant):\n\n"
        "Plant structure (primary):\n"
        "1. Main stem shape: curvature (straight, bent, S-curved), lean direction, "
        "thickness\n"
        "2. Branching pattern: count of primary stems/side-shoots, their angles "
        "(upward, horizontal, drooping)\n"
        "3. Branch height: where the first major side-shoot emerges (low, middle, or "
        "high on the main stem)\n"
        "4. Canopy shape: overall plant shape (narrow/columnar, bushy/spreading, "
        "asymmetric, leaning to one side)\n\n"
        "Support structure (secondary):\n"
        "5. Stake or cage: type (stake, cage, trellis) and its position/angle relative "
        "to the main stem\n"
        "6. Ties/twine/trellis wire: position of horizontal ties or wires, and how the "
        "plant is attached to its support\n"
        "7. Nearby poles/stakes: other supports visible near the plant, and their "
        "position relative to it\n\n"
        "Context (use with caution — can repeat across neighboring plants in a row):\n"
        "8. Row/plot position: whether the plant is at the edge, middle, or end of a row\n"
        "9. Neighbors: distinctive features of adjacent plants\n"
        "10. Background: permanent landmarks (fences, paths, buildings, irrigation lines, "
        "plot markers)\n\n"
        "Ignore these (they change with DAP):\n"
        "- Leaf presence, count, or color\n"
        "- Fruit presence, color, ripeness, or quantity\n"
        "- Flower presence\n"
        "- Overall plant size or greenness\n\n"
        "IMPORTANT — do not read any text: some images may have a plot code, tag, sticker, "
        "or handwritten/printed label burned into the frame (e.g. something like "
        "\"R3T3-b\"). This text is metadata for the dataset, not a visual identity cue, and "
        "reading it defeats the purpose of this test. You MUST ignore any such text "
        "entirely — do not transcribe it, do not compare it across images, and do not let "
        "it influence your answer. Base your decision only on the plant and support "
        "structure itself, as if all such text were blacked out.\n\n"
        "Your task: determine which ONE candidate image shows the SAME individual plant "
        "as the query image.\n\n"
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        f'  "matching_candidate_index": <integer, 1-{num_candidates}, the candidate number '
        "that shows the same plant as the query image>,\n"
        '  "confidence": <float, 0.0-1.0, how confident you are that this match is correct>,\n'
        '  "reasoning": <string, one or two sentences citing ONLY the structural features '
        "you used (never any text/label seen in the images), and why the other candidates "
        "were ruled out>\n"
        "}"
    )


def cross_dap_reid_schema(num_candidates: int) -> dict:
    """JSON schema for the cross-DAP re-identification response, for use with Ollama's `format` param."""
    return {
        "type": "object",
        "properties": {
            "matching_candidate_index": {
                "type": "integer",
                "enum": list(range(1, num_candidates + 1)),
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
        },
        "required": ["matching_candidate_index", "confidence", "reasoning"],
    }


def tomato_ripeness_transition_prompt(earlier_dap, later_dap) -> str:
    """Prompt asking the model to classify how a tomato plant's ripeness changed
    between two DAP snapshots of the same plant.

    Assumes the model is given exactly 2 images in order: Image 1 at `earlier_dap`
    (the smaller DAP value, chronologically first), Image 2 at `later_dap`.
    """
    return (
        "You are an expert agronomist analyzing the ripening progression of a "
        "tomato plant photographed at two different times.\n\n"
        f"Image 1 was taken at {earlier_dap} DAP (Days After Planting).\n"
        f"Image 2 was taken at {later_dap} DAP — a later point in the same plant's "
        "life than Image 1.\n\n"
        "Ripeness stage vision attributes:\n"
        "- Raw: fruit is entirely or predominantly green, with little to no red, "
        "orange, or yellow coloring.\n"
        "- Semi-ripe: fruit is transitioning color, showing yellow or orangish-red, "
        "or a mix of red and green — it has started to ripen but is not yet fully "
        "red.\n"
        "- Ripe: fruit is fully or almost fully red, showing no significant green "
        "or yellow remaining.\n\n"
        "For each image, judge the plant's OVERALL/predominant ripeness stage (the "
        "stage shown by most of the visible fruit), not any single fruit in "
        "isolation.\n\n"
        "Transition indicators:\n"
        "1. Raw to Semi-ripe: green coloring gives way to yellow/orange patches; "
        "fruit begins turning.\n"
        "2. Semi-ripe to Ripe: yellow/orange patches are replaced by more complete "
        "red coloring.\n"
        "3. Raw to Ripe (skipped stage): a large jump straight from green to red, "
        "with no intermediate turning color visible.\n"
        "4. No transition: both images show the same predominant ripeness stage.\n"
        "5. Regression: Image 2 looks LESS ripe overall than Image 1 (e.g. the "
        "riper fruit visible in Image 1 has since been harvested, leaving mostly "
        "younger green fruit, or the images are not actually in chronological "
        "order).\n\n"
        "Your task: determine the predominant ripeness stage in each image, then "
        "classify the transition from Image 1 to Image 2 using ONE of the five "
        "transition types above.\n\n"
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        '  "earlier_stage": <string, one of "Raw", "Semi-ripe", "Ripe" — the '
        "predominant stage in Image 1>,\n"
        '  "later_stage": <string, one of "Raw", "Semi-ripe", "Ripe" — the '
        "predominant stage in Image 2>,\n"
        '  "transition_type": <string, one of "Raw to Semi-ripe", "Semi-ripe to '
        'Ripe", "Raw to Ripe (skipped stage)", "No transition", "Regression">,\n'
        '  "reasoning": <string, one or two sentences citing the specific '
        "color/coverage evidence you used in each image>\n"
        "}"
    )


def tomato_growth_stage_prompt_zeroshot() -> str:
    """Zero-shot (basic) counterpart of tomato_growth_stage_prompt_AG: the task and the
    answer options only, with no visual attributes to examine. Uses tomato_growth_stage_schema.
    """
    return (
        "Look at this tomato plant image and classify its growth stage.\n\n"
        "Options (choose one, or two if the plant is clearly in two stages at once):\n"
        "1. Side shooting\n"
        "2. Flowering\n"
        "3. Fruit development\n"
        "4. Fruit ripening\n"
        "5. Senescence\n\n"
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        '  "growth_stages": [\n'
        "    {\n"
        '      "growth_stage_number": <integer, 1-5>,\n'
        '      "growth_stage_name": <string, one of "Side shooting", "Flowering", '
        '"Fruit development", "Fruit ripening", "Senescence">\n'
        "    }\n"
        "    // one entry, or two entries if two stages are both visible\n"
        "  ],\n"
        '  "reasoning": <string, one or two sentences explaining your answer>\n'
        "}"
    )


def tomato_ripeness_count_prompt_zeroshot() -> str:
    """Zero-shot (basic) counterpart of tomato_ripeness_count_prompt: no ripeness color
    definitions or counting rules. Uses tomato_ripeness_count_schema.
    """
    return (
        "Count the tomatoes visible in this image and report how many are raw, "
        "semi-ripe, and ripe.\n\n"
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        '  "raw_count": <integer>,\n'
        '  "semi_ripe_count": <integer>,\n'
        '  "ripe_count": <integer>,\n'
        '  "total_count": <integer, sum of the three counts above>,\n'
        '  "reasoning": <string, one or two sentences explaining your answer>\n'
        "}"
    )


def cross_dap_reid_prompt_zeroshot(num_candidates: int) -> str:
    """Zero-shot (basic) counterpart of cross_dap_reid_prompt: no structural features to
    compare and no list of DAP-variant cues to ignore. Uses cross_dap_reid_schema.

    Assumes the same image order as cross_dap_reid_prompt: the query image first,
    followed by `num_candidates` candidates.
    """
    candidate_lines = "\n".join(
        f"  - Candidate {i}: Image {i + 1}" for i in range(1, num_candidates + 1)
    )
    return (
        "You are matching tomato plants photographed at different days after planting "
        "(DAP). The same plant looks different at different DAP because it grows, "
        "flowers, and sets fruit, but it is still the same individual plant.\n\n"
        f"You are given {num_candidates + 1} images, in this order:\n"
        "  - Image 1: the QUERY image, showing one specific tomato plant.\n"
        f"{candidate_lines}\n\n"
        "Identify which ONE candidate shows the same plant as the query. Ignore any "
        "text, labels, or codes written in the images.\n\n"
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        f'  "matching_candidate_index": <integer, 1-{num_candidates}>,\n'
        '  "confidence": <float, 0.0-1.0, how confident you are that this match is correct>,\n'
        '  "reasoning": <string, one or two sentences explaining your answer>\n'
        "}"
    )


def tomato_ripeness_transition_prompt_zeroshot(earlier_dap, later_dap) -> str:
    """Zero-shot (basic) counterpart of tomato_ripeness_transition_prompt: the answer
    options only, with no ripeness attributes or transition indicators. Uses
    tomato_ripeness_transition_schema.

    Assumes the same image order: Image 1 at `earlier_dap`, Image 2 at `later_dap`.
    """
    return (
        f"Image 1 and Image 2 show the same tomato plant at {earlier_dap} and "
        f"{later_dap} DAP (Days After Planting). What ripening transition occurred "
        "from Image 1 to Image 2?\n\n"
        "Ripeness stages: Raw, Semi-ripe, Ripe.\n"
        "Transition options:\n"
        "1. Raw to Semi-ripe\n"
        "2. Semi-ripe to Ripe\n"
        "3. Raw to Ripe (skipped stage)\n"
        "4. No transition\n"
        "5. Regression\n\n"
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        '  "earlier_stage": <string, one of "Raw", "Semi-ripe", "Ripe" — the '
        "predominant stage in Image 1>,\n"
        '  "later_stage": <string, one of "Raw", "Semi-ripe", "Ripe" — the '
        "predominant stage in Image 2>,\n"
        '  "transition_type": <string, one of "Raw to Semi-ripe", "Semi-ripe to '
        'Ripe", "Raw to Ripe (skipped stage)", "No transition", "Regression">,\n'
        '  "reasoning": <string, one or two sentences explaining your answer>\n'
        "}"
    )


def tomato_ripeness_transition_schema() -> dict:
    """JSON schema for the ripeness transition response, for use with Ollama's `format` param."""
    stage_names = ["Raw", "Semi-ripe", "Ripe"]
    transition_types = [
        "Raw to Semi-ripe",
        "Semi-ripe to Ripe",
        "Raw to Ripe (skipped stage)",
        "No transition",
        "Regression",
    ]
    return {
        "type": "object",
        "properties": {
            "earlier_stage": {"type": "string", "enum": stage_names},
            "later_stage": {"type": "string", "enum": stage_names},
            "transition_type": {"type": "string", "enum": transition_types},
            "reasoning": {"type": "string"},
        },
        "required": ["earlier_stage", "later_stage", "transition_type", "reasoning"],
    }


GROWTH_STAGE_NAMES = [
    "Side shooting",
    "Flowering",
    "Fruit development",
    "Fruit ripening",
    "Senescence",
]

def growth_stage_set_name(stages) -> str:
    """Canonical name of a set of growth stages: the stage names in developmental order
    joined with " and ", e.g. {"Fruit development", "Flowering"} -> "Flowering and Fruit
    development". Accepts stage names or stage numbers (1-5)."""
    names = {GROWTH_STAGE_NAMES[int(s) - 1] if str(s).strip().isdigit() else str(s).strip() for s in stages}
    return " and ".join(sorted(names, key=GROWTH_STAGE_NAMES.index))


def growth_transition_name(earlier_stages, later_stages) -> str:
    """Canonical Task 5 transition class, e.g. ([2], [2, 3]) ->
    "Flowering to Flowering and Fruit development". Used for both model answers and
    ground truth so that they compare exactly."""
    return f"{growth_stage_set_name(earlier_stages)} to {growth_stage_set_name(later_stages)}"


def _growth_transition_response_format() -> str:
    """JSON response description shared by the AGP and zero-shot growth transition prompts."""
    names = ", ".join(f'"{n}"' for n in GROWTH_STAGE_NAMES)
    return (
        "Respond with ONLY a single JSON object, no other text, matching this exact schema:\n"
        "{\n"
        f'  "earlier_growth_stages": <array of one or two strings, each one of {names} '
        "— the stage(s) visible in Image 1>,\n"
        f'  "later_growth_stages": <array of one or two strings, each one of {names} '
        "— the stage(s) visible in Image 2>,\n"
        '  "transition": <string, the transition from Image 1 to Image 2, written as '
        '"<Image 1 stages> to <Image 2 stages>", e.g. "Flowering to Fruit development">,\n'
        '  "reasoning": <string, one or two sentences citing the specific visual evidence '
        "for the stage(s) in each image>\n"
        "}"
    )


_TRANSITION_FORMAT_RULES = (
    "Write the transition as \"<Image 1 stages> to <Image 2 stages>\", using exactly the "
    "stage(s) you reported for each image. When an image shows two stages, join them with "
    "\"and\" in developmental order (Side shooting, Flowering, Fruit development, Fruit "
    "ripening, Senescence). Write the stages out even when nothing has changed. Examples:\n"
    "No change between the images:\n"
    "- Flowering in both images: \"Flowering to Flowering\"\n"
    "- Side shooting in both images: \"Side shooting to Side shooting\"\n"
    "- Flowering and Fruit development in both images: "
    "\"Flowering and Fruit development to Flowering and Fruit development\"\n"
    "Progression to the next stage:\n"
    "- Side shooting to Flowering: \"Side shooting to Flowering\"\n"
    "- Flowering to Fruit development: \"Flowering to Fruit development\"\n"
    "- Fruit ripening to Senescence: \"Fruit ripening to Senescence\"\n"
    "One image showing two stages:\n"
    "- Flowering in Image 1, Flowering and Fruit development in Image 2: "
    "\"Flowering to Flowering and Fruit development\"\n"
    "- Flowering and Fruit development in Image 1, Fruit development in Image 2: "
    "\"Flowering and Fruit development to Fruit development\"\n"
    "Both images showing two stages:\n"
    "- \"Fruit development and Fruit ripening to Fruit ripening and Senescence\"\n"
    "Larger jumps (a stage skipped between the visits):\n"
    "- Flowering to Fruit ripening: \"Flowering to Fruit ripening\"\n"
    "- Side shooting to Fruit development: \"Side shooting to Fruit development\"\n"
    "Image 2 looking less advanced (e.g. ripe fruit harvested):\n"
    "- \"Fruit ripening to Fruit development\"\n\n"
)


def tomato_growth_transition_prompt(earlier_dap, later_dap) -> str:
    """Attribute-guided prompt for Task 5: classify how a tomato plant's growth stage
    changed between two DAP snapshots of the same plant.

    Assumes the model is given exactly 2 images in order: Image 1 at `earlier_dap`
    (the smaller DAP value, chronologically first), Image 2 at `later_dap`.
    """
    return (
        "You are an expert agronomist analyzing the growth progression of a tomato "
        "plant photographed at two different times.\n\n"
        f"Image 1 was taken at {earlier_dap} DAP (Days After Planting).\n"
        f"Image 2 was taken at {later_dap} DAP — a later point in the same plant's "
        "life than Image 1.\n\n"
        "Growth stages, in developmental order, with their visual attributes:\n"
        "1. Side shooting: vegetative growth with side shoots (suckers) emerging from "
        "the leaf axils; no flowers or fruit yet.\n"
        "2. Flowering: open yellow flowers or flower clusters (trusses) are visible.\n"
        "3. Fruit development: green, unripe fruit is visible and growing.\n"
        "4. Fruit ripening: fruit is colouring — turning yellow, orange or red.\n"
        "5. Senescence: the majority of the foliage is yellowing, browning, wilting or "
        "drying.\n\n"
        "For each image, identify the growth stage. Report two stages only if the plant "
        "clearly shows distinct visual evidence for both at once (for example, flowers and "
        "green fruit together); otherwise report one. Judge each image only by what is "
        "visible in it, not by its DAP value. The later image may show the same stage as "
        "the earlier one, or even an earlier-looking stage (for example, if ripe fruit has "
        "been harvested).\n\n"
        "Then state the growth stage transition from Image 1 to Image 2. "
        + _TRANSITION_FORMAT_RULES
        + _growth_transition_response_format()
    )


def tomato_growth_transition_prompt_zeroshot(earlier_dap, later_dap) -> str:
    """Zero-shot (basic) counterpart of tomato_growth_transition_prompt: the stage names
    and the transition format only, with no visual attributes. Uses
    tomato_growth_transition_schema.

    Assumes the same image order: Image 1 at `earlier_dap`, Image 2 at `later_dap`.
    """
    return (
        f"Image 1 and Image 2 show the same tomato plant at different DAP (Days After Planting). Identify the growth stage in each "
        "image (one stage, or two if the plant is clearly in two stages at once), and "
        "what growth stage transition occurred from Image 1 to Image 2.\n\n"
        "Growth stages: 1. Side shooting, 2. Flowering, 3. Fruit development, "
        "4. Fruit ripening, 5. Senescence.\n\n"
        + _TRANSITION_FORMAT_RULES
        + _growth_transition_response_format()
    )


def tomato_growth_transition_schema() -> dict:
    """JSON schema for the growth transition response, for use with Ollama's `format` param."""
    stage_list = {
        "type": "array",
        "items": {"type": "string", "enum": GROWTH_STAGE_NAMES},
        "minItems": 1,
        "maxItems": 2,
    }
    return {
        "type": "object",
        "properties": {
            "earlier_growth_stages": stage_list,
            "later_growth_stages": stage_list,
            "transition": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["earlier_growth_stages", "later_growth_stages", "transition", "reasoning"],
    }
