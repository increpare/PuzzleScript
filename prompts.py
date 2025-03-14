comments_note = " Any comments must be in the form `(comment)`."
full_game_formatting_prompt = (
    """Do not include more than 5 levels. Return your code in full, inside a ```plaintext code block."""
    # + comments_note
)
game_gen_system_prompt = (
    "You are a creative and resourceful indie puzzle game designer, familiar with the PuzzleScript game description language. "
    """Recall that comments in PuzzleScript are enclosed in parentheses. (E.g. this is a comment.) """
    # f"""Here are the docs: {open('all_documentation.txt', 'r').read()}\n"""
)
fewshow_examples_prompt = (
    "Here are some example games, for inspiration (do not reproduce these games exactly):"""
)
gen_game_prompt = (
    """Output the code for a complete PuzzleScript game. """
    """Do not include any sound effects for now. """
    """The game should be unique and slightly unconventional, with inventive mechanics. """
    """{from_idea_prompt} {cot_prompt}"""
    + full_game_formatting_prompt
)
gen_game_from_idea_prompt = (
    """Create a simplified `demake` of the following game idea in PuzzleScript: {game_idea}. {cot_prompt}"""
    + full_game_formatting_prompt
)
from_idea_repair_prompt = (
    """We are trying to create the following game: {game_idea}. """
)
cot_prompt = (
    """First, reason about your task and determine the best plan of action. Then, write your code. """
)
game_mutate_prompt = (
    """Consider the code for the following PuzzleScript game:\n\n{parents}\n\n"""
    """Create a variation on this game, making it more complex. {cot_prompt}"""
    + full_game_formatting_prompt
)
game_crossover_prompt = (
    """Consider the code for the following PuzzleScript games:\n\n{parents}\n\n"""
    """Create a new, more complex game by combining elements of these games. {cot_prompt}"""
    + full_game_formatting_prompt
)
game_compile_repair_prompt = (
    """{from_idea_repair_prompt}"""
    """The following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """produced the following console output in the online PuzzleScript editor:\n```\n{console_text}\n```\n"""
    """{lark_error_prompt}"""
    """Return a repaired version of the code that addresses these errors. {cot_prompt}"""
    + full_game_formatting_prompt
)
from_idea_prompt = """The game should be a simplified `demake` of the following game idea: {game_idea}"""
game_solvability_repair_prompt = (
    """{from_idea_repair_prompt}"""
    """The following PuzzleScript game code:\n```plaintext\n{code}\n```\n"""
    """compiled, but a solvability check returned the following:\n```\n{solver_text}\n```\n"""
    """Return a repaired version of the code that addresses these errors. {cot_prompt}"""
    + full_game_formatting_prompt
)
# plan_game_prompt = (
#     "Generate a plan for a PuzzleScript game. Describe the game's story/theme, the necessary sprites, "
#     "the mechanics, and the levels that are needed to complete your vision. "
#     "Try to come up with a novel idea, which has not been done before, but which is still feasible "
#     "to implement in PuzzleScript. "
# )
plan_game_prompt = (
    "Generate a high-level plan for a PuzzleScript game. {from_idea_prompt} Think about the necessary objects, "
    "game mechanics and levels that will be needed to complete your vision."
)
gen_sprites_prompt = (
    "Consider the following PuzzleScript game development plan:\n```\n{game_plan}\n```\n"
    ". Select "
    "or generate "
    "the full set of sprites that will be needed to implement this plan. "
    "Here is the existing library of sprites which you may draw from: {sprites_library}.\n\n"
    "Output your response as a list of sprites in a ```plaintext code block, beginning with a header of the form:\n\n"
    "========\nOBJECTS\n========\n\n"
    "To generate a new sprite, define it like this:\n\n"
    "Player\n"
    "black orange white blue\n"
    ".000.\n"
    ".111.\n"
    "22222\n"
    ".333.\n"
    ".3.3.\n\n"
    ", where the first line is the sprite's name, the second line is the colors used in the sprite, "
    "and the following lines are the sprite's pixels, in a 5x54 grid, with indices (starting at 0), "
    "referring to the colors in the second line. "
    "(Colors can also be defined as hex codes, like #FF0000.) "
    "Note that you should favor re-using existing sprites. "
    "To select an existing sprite, simply list their names, like this:\n\n"
    "Player\n\n"
    # "ONLY list sprites that exist in the library above! "
    "After defining the sprites, create a legend that maps sprite names to single-character shorthands, like this:\n\n"
    "========\nLEGEND\n========\n\n"
    "P = Player\n\n"
)
gen_rules_prompt = (
    "Consider the following PuzzleScript game development plan:\n```\n{game_plan}\n```\n"
    ", with the following objects:\n\n{object_legend}\n\n"
    ". Generate the COLLISIONLAYERS, RULES, and WINCONDITIONS to implement the game logic. "
    "Format your output in a single ```plaintext code block, "
    "with each section underneath a header of the form, e.g.:\n"
    "========\nCOLLISIONLAYERS\n========\n"
    + comments_note
)
gen_levels_prompt = (
    "Consider the following PuzzleScript game development plan:\n```\n{game_plan}\n```\n"
    ", and the following partially complete PuzzleScript code:\n```plaintext\n{code}\n```\n"
    ". Generate a few levels to complete the design plan. "
    "Output your response as a list of levels in a ```plaintext code block. "
    "(You do not need to include the entire code, just the levels.)"
)
gen_finalize_prompt = (
    "Consider the following PuzzleScript game development plan:\n```\n{game_plan}\n```\n"
    ", and the following partially complete PuzzleScript code:\n```plaintext\n{code}\n```\n"
    ". Complete the code to implement the game. "
    + full_game_formatting_prompt
)
