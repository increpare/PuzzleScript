
var errormessage_testdata = [
    [
        "Background missing",
        [{file:"test_scripts/background_missing.txt"}, ["Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!", "Seriously, you have to define something to be the background."], 2]
    ],
    [
        `Player missing`,
        [{file:"test_scripts/player_missing.txt"}, ["Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!", "Error, didn't find any object called player, either in the objects section, or the legends section."], 2]
    ],
    [
        "Sprites must be 5 wide and 5 high.",
        [{file:"test_scripts/sprites_must_be_5_wide_and_5_high.txt"}, ["line 34 : Sprites must be 5 wide and 5 high.", "line 29 : Sprite graphics must be 5 wide and 5 high exactly."], 0]
    ],
    [
        `No levels found`,
        ["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\n\nTarget\nDarkBlue\n.....\n.000.\n.0.0.\n.000.\n.....\n\nWall\nBROWN DARKBROWN\n00010\n11111\n01000\n11111\n00010\n\nPlayer\nBlack Orange White Blue\n.000.\n.111.\n22222\n.333.\n.3.3.\n\nCrate\nOrange Yellow\n00000\n0...0\n0...0\n0...0\n00000\n\n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\n@ = Crate and Target\nO = Target\n\n\n=======\nSOUNDS\n=======\n\nCrate MOVE 36772507\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES\n======\n\n[ > Player | Crate ] -> [ > Player | > Crate ]\n\n==============\nWINCONDITIONS\n==============\n\nAll Target on Crate\n\n=======\nLEVELS\n=======\n\n", ["No levels found. Add some levels!"], 1]
    ],
    [
        "A rule has to have an arrow in it.",
        [{file:"test_scripts/a_rule_has_to_have_an_arrow_in_it.txt"}, ["line 78 : A rule has to have an arrow in it. There's no arrow here! Consider reading up about rules - you're clearly doing something weird", "line 78 : Error, when specifying a rule, the number of matches (square bracketed bits) on the left hand side of the arrow must equal the number on the right"], 1]
    ],
    [
        `each pattern to match on the left must have a corresponding pattern on the right of equal length`,
        [{file:"test_scripts/each_pattern_to_match_on_the_left_must_have_a_corresponding.txt"}, ["line 140 : In a rule, each pattern to match on the left must have a corresponding pattern on the right of equal length (number of cells)."], 1]
    ],
    [
        `An ellipsis on the left must be matched by one in the corresponding place on the right.`,
        [{file:"test_scripts/an_ellipsis_on_the_left_must_be_matched_by_one_in_the_corres.txt"}, ["line 140 : An ellipsis on the left must be matched by one in the corresponding place on the right."], 1]
    ],
    [
        `You can't have anything in with an ellipsis. Sorry.`,
        [{file:"test_scripts/you_can_t_have_anything_in_with_an_ellipsis_sorry.txt"}, ["line 140 : You can't have anything in with an ellipsis. Sorry."], 1]
    ],
    [
        `Single-character keyword legend synonym is allowed`,
        [{file:"test_scripts/you_named_an_object_but_this_is_a_keyword_don_t_do_that.txt"}, [], 0]
    ],
    [
        "incorrect sound declaration",
        [{file:"test_scripts/incorrect_sound_declaration.txt"}, ["line 76 : unexpected sound token \"boop\"."], 1]
    ],
    [
        `Unrecognised stuff in the prelude`,
        [{file:"test_scripts/unrecognised_stuff_in_the_prelude.txt"}, ["line 4 : Unrecognised stuff in the prelude."], 1]
    ],
    [
        `Game title is too long to fit on screen; truncating to fit.`,
        [{file:"test_scripts/game_title_is_too_long_to_fit_on_screen_truncating_to_fit.txt"}, ["line 1 : Game title is too long to fit on screen; truncating to fit."], 0]
    ],
    [
        `You probably meant to put a space after 'message' innit.`,
        [{file:"test_scripts/you_probably_meant_to_put_a_space_after_message_innit.txt"}, ["line 93 : You probably meant to put a space after 'message' innit. That's ok, I'll still interpret it as a message, but you probably want to put a space there."], 0]
    ],
    [
        `Block Faker`,
        [{file:"test_scripts/block_faker.txt"}, [], 0]
    ],
    [
        `Atlas Shrank`,
        [{file:"test_scripts/atlas_shrank_2.txt"}, [], 0]
    ],
    [
        `Simple Action Example`,
        [{file:"test_scripts/simple_action_example.txt"}, [], 0]
    ],
    [
        `actiontest.txt`,
        [{file:"test_scripts/actiontest_txt.txt"}, [], 0]
    ],
    [
        `zenpuzzlegarden`,
        [{file:"test_scripts/zen_puzzle_garden.txt"}, [], 0]
    ],
    [
        `wrappingrecipe`,
        ["title Wrapping Recipe\nauthor increpare\nhomepage http://www.puzzlescript.net\n\n========\nOBJECTS\n========\n\nWrap\nBlack\n\nBackground\nWhite\n\nPlayer\nBlue\n\n=======\nLEGEND\n=======\n\n. = Background\n~ = Wrap\n@ = Player\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nWrap\nPlayer\n\n======\nRULES\n======\n\n[ Wrap | < Player | ... | | Wrap ] -> [ Wrap | | ... | Player | Wrap ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n~~~~~~~\n~.....~\n~.....~\n~..@..~\n~.....~\n~.....~\n~~~~~~~\n\n", [], 0]
    ],
    [
        `wordgame`,
        [{file:"test_scripts/wordgame.txt"}, [], 0]
    ],
    [
        `whaleworld`,
        [{file:"test_scripts/whaleworld.txt"}, [], 0]
    ],
    [
        `twolittlecrates4`,
        [{file:"test_scripts/twolittlecrates4.txt"}, ["line 68 : Name \"O\" already in use (on line line 51)."], 1]
    ],
    [
        `twolittlecrates3`,
        [{file:"test_scripts/twolittlecrates3.txt"}, ["line 68 : Name \"O\" already in use (on line line 51)."], 1]
    ],
    [
        `twolittlecrates2`,
        [{file:"test_scripts/twolittlecrates2.txt"}, [], 0]
    ],
    [
        `twolittlecrates1`,
        [{file:"test_scripts/twolittlecrates1.txt"}, [], 0]
    ],
    [
        `tunnel rat`,
        [{file:"test_scripts/tunnel_rat_2.txt"}, [], 0]
    ],
    [
        `tiny treasure hunt`,
        [{file:"test_scripts/tiny_treasure_hunt_2.txt"}, [], 0]
    ],
    [
        `threes`,
        [{file:"test_scripts/threes_2.txt"}, [], 0]
    ],
    [
        `candy scroll saga`,
        [{file:"test_scripts/candy_scroll_saga.txt"}, [], 0]
    ],
    [
        `take heart lass`,
        [{file:"test_scripts/take_heart_lass_2.txt"}, [], 0]
    ],
    [
        `sumo`,
        ["title Multiple Patterns\nauthor increpare\nhomepage www.puzzlescript.net\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\n\nWall\nBROWN DARKBROWN\n00010\n11111\n01000\n11111\n00010\n\nPlayer\nBlack Orange White Blue\n.000.\n.111.\n22222\n.333.\n.3.3.\n\nSumo\nPink White\n..0..\n00000\n.000.\n.111.\n.0.0.\n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\nS = Sumo\n\n=======\nSOUNDS\n=======\n\nSumo MOVE 48878907\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Wall, Sumo\n\n======\nRULES\n======\n\n[ > Player ] [ Sumo ] -> [ > Player ] [ > Sumo ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n########\n#......#\n#......#\n#.P..S.#\n#......#\n#......#\n########\n\n", [], 0]
    ],
    [
        `stick candy puzzle saga`,
        [{file:"test_scripts/stick_candy_puzzle_saga.txt"}, [], 0]
    ],
    [
        `sokobond demake`,
        [{file:"test_scripts/sokobond_demake_2.txt"}, [], 0]
    ],
    [
        `sokoban sticky`,
        [{file:"test_scripts/sokoban_sticky.txt"}, [], 0]
    ],
    [
        `sokoban match 3`,
        [{file:"test_scripts/sokoban_match_3.txt"}, [], 0]
    ],
    [
        `sokoban horizontal`,
        [{file:"test_scripts/sokoban_horizontal.txt"}, [], 0]
    ],
    [
        `sokoban eyeball`,
        [{file:"test_scripts/sokoban_eyeball.txt"}, [], 0]
    ],
    [
        `sokoban basic`,
        [{file:"test_scripts/undoing_reset_undoes_two_steps_not_one_453.txt"}, [], 0]
    ],
    [
        `sok7`,
        [{file:"test_scripts/sok7_2.txt"}, [], 0]
    ],
    [
        `smother`,
        [{file:"test_scripts/smother_2.txt"}, ["line 168 : Object \"PLAYER\" included in multiple collision layers ( layers #4, #5 ). You should fix this!", "line 168 : Object \"WALL\" included in multiple collision layers ( layers #4, #5 ). You should fix this!"], 0]
    ],
    [
        `slidings`,
        [{file:"test_scripts/slidings.txt"}, [], 0]
    ],
    [
        `several rigid bodies`,
        [{file:"test_scripts/several_rigid_bodies.txt"}, [], 0]
    ],
    [
        `scriptcross`,
        [{file:"test_scripts/scriptcross.txt"}, [], 0]
    ],
    [
        `robotarm`,
        [{file:"test_scripts/robotarm.txt"}, [], 0]
    ],
    [
        `river puzzle`,
        [{file:"test_scripts/river_puzzle.txt"}, [], 0]
    ],
    [
        `rigid fail`,
        [{file:"test_scripts/rigid_fail.txt"}, [], 0]
    ],
    [
        `rigid scott`,
        [{file:"test_scripts/rigid_scott.txt"}, [], 0]
    ],
    [
        `rigid one object`,
        [{file:"test_scripts/rigid_one_object.txt"}, [], 0]
    ],
    [
        `rigid and little`,
        [{file:"test_scripts/rigid_and_little.txt"}, [], 0]
    ],
    [
        `randomspawner`,
        [{file:"test_scripts/randomspawner.txt"}, [], 0]
    ],
    [
        `randomrobots`,
        ["title Random Robots\nauthor increpare\nhomepage www.puzzlescript.net\n\n========\nOBJECTS\n========\n\n\nBackground (special, automatically placed in its own layer)\nGREEN darkgreen\n11111\n01111\n11101\n11111\n10111\n\n\nPlayer (Required by game)\nPINK\n..0..\n00000\n..0..\n.000.\n.0.0.\n\n\nRobot\ngray black red\n.000.\n.200.\n...0.\n00000\n01010\n\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = Robot\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Robot\n\n======\nRULES\n======\n\n[ stationary Robot ] -> [randomDir Robot]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n...........\n.....*.....\n...........\n.....*.....\n...........\n.....*.....\n..P........\n.....*.....\n...........\n\n", [], 0]
    ],
    [
        `push`,
        [{file:"test_scripts/push_2.txt"}, [], 0]
    ],
    [
        `puzzles`,
        [{file:"test_scripts/puzzles.txt"}, [], 0]
    ],
    [
        `octat`,
        [{file:"test_scripts/octat.txt"}, [], 0]
    ],
    [
        "Ponies Jumping Synchronously",
        [{file:"test_scripts/ponies.txt"}, ["line 349 : You're trying to close a comment here, but I can't find any opening bracket to match it? [This is highly suspicious; you probably want to fix it.]", "line 343 : Encountered an unexpected \"->\" inside square brackets. It's used to separate states, it has no place inside them >:| .", "line 343 : Multiple opening brackets without closing brackets. Something fishy here. Every '[' has to be closed by a ']', and you can't nest them.", "line 343 : Error, when specifying a rule, the number of matches (square bracketed bits) on the left hand side of the arrow must equal the number on the right"], 0]
    ],
    [
        `notsnake`,
        ["title Notsnake\nauthor Terry Cavanagh\nhomepage www.distractionware.com\n\ncolor_palette c64\nrequire_player_movement\n\n========\nOBJECTS\n========\n\nBackground\nblue\n\nPlayer\nlightblue\n\nTrail\npurple\n\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nO = Trail\n\n=======\nSOUNDS\n=======\n\nstartgame 50287705\nsfx0 660507\nsfx1 41058704\nendgame 74955903\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTrail\nPlayer\n\n======\nRULES\n======\n\n[ > Player | No Trail ] -> [ Trail | Player ] sfx0\n[ > Player | Trail ] -> [ | Player ] sfx1\n\n==============\nWINCONDITIONS\n==============\n\nNo Trail\n\n=======\nLEVELS\n=======\n\nOOOOOOOO\nOOOOOOOO\nO..POO.O\nOOOOOOOO\nOOOOOOOO\n\n", [], 0]
    ],
    [
        `nekopuzzle`,
        [{file:"test_scripts/nekopuzzle.txt"}, [], 0]
    ],
    [
        `naughtysprite`,
        [{file:"test_scripts/naughtysprite.txt"}, [], 0]
    ],
    [
        `modality`,
        [{file:"test_scripts/modality_2.txt"}, [], 0]
    ],
    [
        `midas`,
        [{file:"test_scripts/gallery_midas.txt"}, [], 0]
    ],
    [
        `microban`,
        [{file:"test_scripts/microban.txt"}, [], 0]
    ],
    [
        `mazezezam`,
        [{file:"test_scripts/mazezezam.txt"}, [], 0]
    ],
    [
        `mazetest`,
        [{file:"test_scripts/mazetest.txt"}, [], 0]
    ],
    [
        `manic ammo`,
        [{file:"test_scripts/manic_ammo_2.txt"}, [], 0]
    ],
    [
        `mc escher`,
        [{file:"test_scripts/mc_escher.txt"}, ["line 235 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 236 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 237 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 238 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 239 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 240 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 241 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 242 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 243 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 244 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 245 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 246 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 247 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 248 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 249 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 250 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 251 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 252 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 253 : Maps must be rectangular, yo (In a level, the length of each row must be the same)."], 0]
    ],
    [
        `lunar lockout`,
        [{file:"test_scripts/lunar_lockout.txt"}, [], 0]
    ],
    [
        `love and pieces`,
        [{file:"test_scripts/love_and_pieces_2.txt"}, [], 0]
    ],
    [
        `lime rick`,
        [{file:"test_scripts/lime_rick.txt"}, [], 0]
    ],
    [
        `legend of zokoban`,
        [{file:"test_scripts/legend_of_zokoban_2.txt"}, [], 0]
    ],
    [
        `leftrightnpcs`,
        [{file:"test_scripts/leftrightnpcs.txt"}, [], 0]
    ],
    [
        `led challenge`,
        [{file:"test_scripts/circuit_game.txt"}, [], 0]
    ],
    [
        `my first Kishoutenketsu`,
        [{file:"test_scripts/my_first_kishoutenketsu.txt"}, [], 0]
    ],
    [
        `kettle`,
        [{file:"test_scripts/kettle.txt"}, [], 0]
    ],
    [
        `it dies in the light`,
        [{file:"test_scripts/it_dies_in_the_light_2.txt"}, [], 0]
    ],
    [
        `icycrates`,
        [{file:"test_scripts/icycrates_2.txt"}, [], 0]
    ],
    [
        `icecrates`,
        [{file:"test_scripts/icycrates_2.txt"}, [], 0]
    ],
    [
        `heroes of sokoban 3`,
        [{file:"test_scripts/heroes_of_sokoban_3_2.txt"}, [], 0]
    ],
    [
        `heroes of sokoban 2`,
        [{file:"test_scripts/heroes_of_sokoban_2.txt"}, [], 0]
    ],
    [
        `heroes of sokoban 1`,
        [{file:"test_scripts/heroes_of_sokoban_1.txt"}, [], 0]
    ],
    [
        `gobble rush`,
        [{file:"test_scripts/gobble_rush.txt"}, [], 0]
    ],
    [
        `ebony and ivory`,
        [{file:"test_scripts/ebony_and_ivory.txt"}, [], 0]
    ],
    [
        `easy enigma`,
        [{file:"test_scripts/easy_enigma.txt"}, [], 0]
    ],
    [
        `dungeon janitor`,
        [{file:"test_scripts/dungeon_janitor_2.txt"}, [], 0]
    ],
    [
        `drop swap`,
        [{file:"test_scripts/drop_swap_2.txt"}, ["line 236 : Name \"P\" already in use (on line line 221)."], 1]
    ],
    [
        `cute train`,
        [{file:"test_scripts/cute_train_2.txt"}, [], 0]
    ],
    [
        `cratopia`,
        [{file:"test_scripts/cratopia_2.txt"}, ["line 307 : Object \"EDITMODECURSOR\" included in multiple collision layers ( layers #3, #4 ). You should fix this!", "line 533 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 534 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 535 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 566 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 567 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 568 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 569 : Maps must be rectangular, yo (In a level, the length of each row must be the same).", "line 342 : Invalid syntax. Directions should be placed at the start of a rule."], 0]
    ],
    [
        `contellationz`,
        [{file:"test_scripts/contellationz.txt"}, [], 0]
    ],
    [
        `colour chained`,
        [{file:"test_scripts/colour_chained.txt"}, [], 0]
    ],
    [
        `collapse`,
        [{file:"test_scripts/collapse.txt"}, ["line 178 : Name \"P\" already in use (on line line 129)."], 1]
    ],
    [
        `coin counter`,
        [{file:"test_scripts/coin_counter.txt"}, [], 0]
    ],
    [
        `chaos wizard`,
        [{file:"test_scripts/chaos_wizard_2.txt"}, [], 0]
    ],
    [
        `castle mouse`,
        [{file:"test_scripts/castle_mouse.txt"}, [], 0]
    ],
    [
        `castle closet`,
        [{file:"test_scripts/castle_closet.txt"}, [], 0]
    ],
    [
        `cake monster`,
        [{file:"test_scripts/cake_monster.txt"}, ["line 22 : Was looking for color for object 00000, got \"020\" instead.", "line 22 : Was looking for color for object 00000, got \".3\" instead.", "line 23 : Trying to access color number 0 from the color palette of sprite 00000, but there are only 0 defined in it.", "line 24 : Trying to access color number 0 from the color palette of sprite 00000, but there are only 0 defined in it.", "line 25 : Trying to access color number 1 from the color palette of sprite 00000, but there are only 0 defined in it.", "line 30 : Object \"00000\" defined multiple times."], 6]
    ],
    [
        `by your side`,
        [{file:"test_scripts/by_your_side_2.txt"}, [], 0]
    ],
    [
        `bouncers`,
        [{file:"test_scripts/bouncers_2.txt"}, [], 0]
    ],
    [
        `cake monsters`,
        [{file:"test_scripts/cake_monsters.txt"}, [], 0]
    ],
    [
        `loop deleter`,
        [{file:"test_scripts/loop_deleter.txt"}, [], 0]
    ],
    [
        `I'm too far gone`,
        [{file:"test_scripts/gallery_game_i_m_too_far_gone.txt"}, [], 0]
    ],
    [
        "Rules with overlapping objects",
        [{file:"test_scripts/rules_with_overlapping_objects.txt"}, ["line 179 : You cannot specify the same object more than once in a single cell (in this case PRESENT occurs multiple times).", "( However, noticing that you're committing this crime with properties, and not being able to help but acknowledge that you may be trying to do something esoteric and clever with the property inference system, I might be brought to suggest that you consider this: you can have multiple equivalent properties with different names. )", "line 185 : You cannot specify the same object more than once in a single cell (in this case PRESENT occurs multiple times).", "line 179 : P0 and P0 can never overlap (ditto for P1/P1, P2/P2, P3/P3 etc.). This rule requires that to happen, so it impossible it will ever trigger.", "line 185 : P0 and P0 can never overlap (ditto for P1/P1, P2/P2, P3/P3 etc.). This rule requires that to happen, so it impossible it will ever trigger."], 5]],
    [
        `ambiguity test (cf #498): right [ vertical Player | vertical Player | ] -> [ | | vertical Player ]`,
        ["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nbaba = player or Background\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\nright [ vertical Player | vertical Player | ] -> [ | | vertical Player ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...pp..\n.......\n", ["line 41 : This rule has an ambiguous movement on the right-hand side, \"vertical\", that can't be inferred from the left-hand side. (either for every ambiguous movement associated to an entity on the right there has to be a corresponding one on the left attached to the same entity, OR, if there's a single occurrence of a particular ambiguous movement on the left, all properties of the same movement attached to the same object on the right are assumed to be the same (or something like that))."], 1]
    ],
    [
        `ambiguity test (cf #498): right [ vertical Player | vertical Player | ] -> [ | | vertical baba ]`,
        ["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nbaba = player or Background\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\nright [ vertical Player | vertical Player | ] -> [ | | vertical baba ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...pp..\n.......\n", ["line 41 : This rule has an ambiguous movement on the right-hand side, \"vertical\", that can't be inferred from the left-hand side. (either for every ambiguous movement associated to an entity on the right there has to be a corresponding one on the left attached to the same entity, OR, if there's a single occurrence of a particular ambiguous movement on the left, all properties of the same movement attached to the same object on the right are assumed to be the same (or something like that)).", "line 41 : This rule has a property on the right-hand side, \"BABA\", that can't be inferred from the left-hand side. (either for every property on the right there has to be a corresponding one on the left in the same cell, OR, if there's a single occurrence of a particular property name on the left, all properties of the same name on the right are assumed to be the same)."], 2]
    ],
    [
        `ambiguity test (cf #498): right [ vertical Player | vertical Player  ] -> [ horizontal player | moving player ]`,
        ["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nbaba = player or Background\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\nright [ vertical Player | vertical Player ] -> [ horizontal player | moving player ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...pp..\n.......\n", ["line 41 : This rule has an ambiguous movement on the right-hand side, \"horizontal\", that can't be inferred from the left-hand side. (either for every ambiguous movement associated to an entity on the right there has to be a corresponding one on the left attached to the same entity, OR, if there's a single occurrence of a particular ambiguous movement on the left, all properties of the same movement attached to the same object on the right are assumed to be the same (or something like that))."], 1]
    ],
    [
        `ambiguity test.  in principle I'd like to support interpreting this (order-preserving disambiguation of aggregate movements), but right now I don't right [ vertical Player | vertical Player | | ] -> [ | |  vertical Player | vertical Player ]`,
        ["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nbaba = player or Background\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\nright [ vertical Player | vertical Player | | ] -> [ || vertical Player | vertical Player ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...pp..\n.......\n", ["line 41 : This rule has an ambiguous movement on the right-hand side, \"vertical\", that can't be inferred from the left-hand side. (either for every ambiguous movement associated to an entity on the right there has to be a corresponding one on the left attached to the same entity, OR, if there's a single occurrence of a particular ambiguous movement on the left, all properties of the same movement attached to the same object on the right are assumed to be the same (or something like that))."], 1]
    ],
    [
        "censored version of NSFW game `Pornography for Beginners` by Holly Gramazio",
        [{file:"test_scripts/gallery_censored_version_of_nsfw_game_pornography_for_beginn.txt"}, ["line 744 : Sprites must be 5 wide and 5 high.", "line 752 : Sprites must be 5 wide and 5 high.", "line 1159 : Object \"BBBNNNPLAYER\" included in multiple collision layers ( layers #4, #6 ). You should fix this!", "line 742 : Sprite graphics must be 5 wide and 5 high exactly.", "line 750 : Sprite graphics must be 5 wide and 5 high exactly."], 0]
    ],
    [
        "crash bug when 'blank project' is selected #738",
        ["========\nOBJECTS\n========\n\n=======\nLEGEND\n=======\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\n======\nRULES\n======\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n", ["Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!", "Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!", "No collision layers defined. All objects need to be in collision layers."], 3]
    ],
    [
        "A rule-group can only be marked random by the opening rule in the group",
        [{file:"test_scripts/a_rule_group_can_only_be_marked_random_by_the_opening_rule_i.txt"}, ["line 84 : A rule-group can only be marked random by the opening rule in the group (aka, a '+' and 'random' can't appear as rule modifiers on the same line). Why? Well, you see \"random\" isn't a property of individual rules, but of whole rule groups. It indicates that a single possible application of some rule from the whole group should be applied at random."], 1]
    ],
    [
        "Cancel appearing in weird places 705",
        [{file:"test_scripts/cancel_appearing_in_weird_places_705.txt"}, ["line 88 : The start of a rule must consist of some number of directions (possibly 0), before the first bracket, specifying in what directions to look (with no direction specified, it applies in all four directions). It seems you've just entered \"CANCEL\".", "line 89 : Commands should only appear at the end of rules, not in or before the pattern-detection/-replacement sections.", "line 90 : Commands should only appear at the end of rules, not in or before the pattern-detection/-replacement sections.", "line 91 : Commands should only appear at the end of rules, not in or before the pattern-detection/-replacement sections."], 2]
    ],
    [
        "for the occasion of the refactor of 'check that no late rule contains movemen' #750",
        [{file:"test_scripts/for_the_occasion_of_the_refactor_of_check_that_no_late_rule.txt"}, ["line 81 : Movements cannot appear in late rules.", "line 86 : Movements cannot appear in late rules."], 2]
    ],
    [
        "X no X causing rulegroup to vanish error message #762",
        [{file:"test_scripts/x_no_x_causing_rulegroup_to_vanish_error_message_762.txt"}, ["line 83 : You cannot specify the same object more than once in a single cell (in this case PLAYER occurs multiple times).", "line 83 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 1]
    ],
    [
        "VEXT EDIT COMPILE TEST",
        [{file:"test_scripts/vext_edit_compile_test.txt"}, ["line 952 : You're repeating the object CRATE1MARK here multiple times on the RHS. This makes no sense. Don't do that.", "line 1134 : This rule has something amounting to \"CURSOR0A NO CURSOR0A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1134 : This rule has something amounting to \"DEAD0A NO DEAD0A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1134 : This rule has something amounting to \"CURSOR0B NO CURSOR0B\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1135 : This rule has something amounting to \"CURSOR0A NO CURSOR0A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1135 : This rule has something amounting to \"DEAD0A NO DEAD0A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1135 : This rule has something amounting to \"CURSOR0B NO CURSOR0B\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1136 : This rule has something amounting to \"CURSOR0A NO CURSOR0A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1136 : This rule has something amounting to \"DEAD0A NO DEAD0A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1136 : This rule has something amounting to \"CURSOR0B NO CURSOR0B\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1137 : This rule has something amounting to \"CURSOR0A NO CURSOR0A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1137 : This rule has something amounting to \"DEAD0A NO DEAD0A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 1137 : This rule has something amounting to \"CURSOR0B NO CURSOR0B\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],
    [
        "Editor crash with opening parenthesis in level section in middle of line #784",
        [{file:"test_scripts/editor_crash_with_opening_parenthesis_in_level_section_in_mi.txt"}, ["line 127 : You're trying to close a comment here, but I can't find any opening bracket to match it? [This is highly suspicious; you probably want to fix it.]"], 0]
    ],
    [
        "#789 1 Lime Rick (recursive player at start of list)",
        [{file:"test_scripts/789_1_lime_rick_recursive_player_at_start_of_list.txt"}, ["line 107 : You can't define object PLAYER in terms of itself!"], 1]
    ],
    [
        "#789 2 Lime Rick (recursive player in middle of list)",
        [{file:"test_scripts/789_2_lime_rick_recursive_player_in_middle_of_list.txt"}, ["line 107 : You can't define object PLAYER in terms of itself!"], 1]
    ],
    [
        "#789 3 Lime Rick (recursive player at end of list)",
        [{file:"test_scripts/789_3_lime_rick_recursive_player_at_end_of_list.txt"}, ["line 107 : You can't define object PLAYER in terms of itself!"], 1]
    ],
    [
        "#789 4 Lime Rick (recursive player solo)",
        [{file:"test_scripts/789_4_lime_rick_recursive_player_solo.txt"}, ["line 107 : You can't define object PLAYER in terms of itself!"], 1]
    ],
    [
        "#779 'sfx0 49518300 destroy' works",
        [{file:"test_scripts/779_sfx0_49518300_destroy_works.txt"}, ["line 67 : I wasn't expecting anything after the sound declaration 142315 on this line, so I don't know what to do with \"MOVE\" here.", "line 75 : I wasn't expecting anything after the sound declaration 213424 on this line, so I don't know what to do with \"SFX1\" here."], 2]
    ],
    [
        "#804 extra closing brackets accepted without comment",
        ["OBJECTS\n\nBackground Player .\nBlack\n\nLEGEND\n\nBad = )Player or Player\n\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\n\nRULES\nWINCONDITIONS\nLEVELS\n\n.\n", ["line 8 : You're trying to close a comment here, but I can't find any opening bracket to match it? [This is highly suspicious; you probably want to fix it.]"], 0]
    ],
    [
        "parsing sections names, not allowing things directly after #805",
        ["OBJECTS.\nRed\n\n", ["line 1 : Only comments should go after a section name (\"OBJECTS\") on a line.", "Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!", "Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!", "No collision layers defined. All objects need to be in collision layers."], 4]
    ],
    [
        "#805 parsing sections names, not allowing things directly after (no error test)",
        [{file:"test_scripts/805_parsing_sections_names_not_allowing_things_directly_afte.txt"}, [], 0]
    ],
    [
        "#805 untitled test (Error producing test)",
        ["Objects are fun to use.\n\nBackground .\nBlack\n\nPlayer P\nGreen\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\nWINCONDITIONS\n\nNo Fun\n\nLEVELS\n\nP...\n", ["line 1 : Only comments should go after a section name (\"OBJECTS\") on a line.", "line 19 : Error in win condition: \"FUN\" is not a valid object name.", "line 19 : Unwelcome term \"fun\" found in win condition. I don't know what I'm supposed to do with this."], 3]
    ],
    [
        "#809 stress-test for legend+comments",
        [{file:"test_scripts/809_stress_test_for_legend_comments.txt"}, ["line 75 : Something bad's happening in the LEGEND", "line 75 : Incorrect format of legend - should be one of \"A = B\", \"A = B or C [ or D ...]\", \"A = B and C [ and D ...]\", but it looks like you have a dangling \"=\"?"], 2]
    ],
    [
        "#808 PLAYER and PLAYER can never overlap, but this rule requires that to happen",
        [{file:"test_scripts/808_player_and_player_can_never_overlap_but_this_rule_requir.txt"}, ["line 59 : Trying to create an aggregate object (something defined in the LEGEND section using AND) with both \"PLAYER\" and \"PLAYER\", which are on the same layer and therefore can't coexist.", "line 83 : PLAYER and PLAYER can never overlap. This rule requires that to happen, so it impossible it will ever trigger."], 2]
    ],
    [
        "#815 problem with dead rule elimination and 'Rules with overlapping objects' test",
        [{file:"test_scripts/815_problem_with_dead_rule_elimination_and_rules_with_overla.txt"}, ["line 171 : You cannot specify the same object more than once in a single cell (in this case PRESENT occurs multiple times).", "( However, noticing that you're committing this crime with properties, and not being able to help but acknowledge that you may be trying to do something esoteric and clever with the property inference system, I might be brought to suggest that you consider this: you can have multiple equivalent properties with different names. )", "line 171 : You cannot specify the same object more than once in a single cell (in this case PLAYER occurs multiple times).", "line 173 : You cannot specify the same object more than once in a single cell (in this case PRESENT occurs multiple times).", "line 173 : You cannot specify the same object more than once in a single cell (in this case PLAYER occurs multiple times).", "line 177 : You cannot specify the same object more than once in a single cell (in this case PRESENT occurs multiple times).", "line 177 : You cannot specify the same object more than once in a single cell (in this case PLAYER occurs multiple times).", "line 171 : P0 and P0 can never overlap (ditto for P1/P1, P2/P2, P3/P3 etc.). This rule requires that to happen, so it impossible it will ever trigger.", "line 173 : P0 and P0 can never overlap (ditto for P1/P1, P2/P2, P3/P3 etc.). This rule requires that to happen, so it impossible it will ever trigger.", "line 177 : P0 and P0 can never overlap (ditto for P1/P1, P2/P2, P3/P3 etc.). This rule requires that to happen, so it impossible it will ever trigger."], 10]],
    [
        "#817 removing youtube stuff",
        [{file:"test_scripts/817_removing_youtube_stuff.txt"}, ["line 6 : Unfortunately, YouTube support hasn't been working properly for a long time - it was always a hack and it hasn't gotten less hacky over time, so I can no longer pretend to support it."], 0]
    ],
    [
        "#819 You can't define an object OR in terms of itself",
        ["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\n========\nOBJECTS\n========\n\nBackground .\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\n\nTarget\nDarkBlue\n.....\n.000.\n.0.0.\n.000.\n.....\n\nWall\nBROWN DARKBROWN\n00010\n11111\n01000\n11111\n00010\n\nPlayer\nBlack Orange White Blue\n.000.\n.111.\n22222\n.333.\n.3.3.\n\nCrate \nOrange\n00000\n0...0\n0...0\n0...0\n00000\n\n\n=======\nLEGEND\n=======\n\nor = background or wall \nand = player and wall\n\n=======\nSOUNDS\n=======\n\nCrate MOVE 36772507\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES\n======\n\n[ or ] -> [ and ]\n\n==============\nWINCONDITIONS\n==============\n\nAll Target on Crate\n\n=======\nLEVELS\n=======\n\n.\n", ["line 56 : Trying to create an aggregate object (something defined in the LEGEND section using AND) with both \"WALL\" and \"PLAYER\", which are on the same layer and therefore can't coexist.", "line 76 : Rule matches object types that can't overlap: \"WALL\" and \"PLAYER\"."], 2]
    ],
    [
        "#820 ditto error message sometimes spits out duplicates",
        [{file:"test_scripts/820_ditto_error_message_sometimes_spits_out_duplicates.txt"}, ["line 61 : Trying to create an aggregate object (something defined in the LEGEND section using AND) with both \"WALL\" and \"CRATE\", which are on the same layer and therefore can't coexist.", "line 81 : WALL and CRATE can never overlap. This rule requires that to happen, so it impossible it will ever trigger."], 2]
    ],
    [
        "#821 people probably should be warned against naming objects the same as sound events (SFX0, cantmove, etc)",
        [{file:"test_scripts/821_people_probably_should_be_warned_against_naming_objects.txt"}, ["line 26 : You named an object \"ACTION\", but this is a keyword. Don't do that!", "line 45 : You named an object \"CANTMOVE\", but this is a keyword. Don't do that!", "line 64 : You named an object \"SFX3\", but this is a keyword. Don't do that!"], 0]
    ],
    [
        "Testing error messages for incorrect semantic in the sounds section",
        [{file:"test_scripts/testing_error_messages_for_incorrect_semantic_in_the_sounds.txt"}, ["line 67 : Ah I was expecting direction or a sound seed here after UP, but I don't know what to make of \"CAT\".", "line 68 : Ah I was expecting a sound seed here after CREATE, but I don't know what to make of \"UP\".", "line 69 : Ah I was expecting a sound seed here after CREATE, but I don't know what to make of \"CAT\"."], 3]
    ],
    [
        "Sound section comment insertion stress test",
        [{file:"test_scripts/sound_section_comment_insertion_stress_test.txt"}, [], 0]
    ],
    [
        "Sound effect already declared message",
        [{file:"test_scripts/sound_effect_already_declared_message.txt"}, ["line 121 : SFX0 already declared."], 0]
    ],
    [
        "#855 Pushing It",
        [{file:"test_scripts/855_pushing_it.txt"}, ["Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!", "Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!", "No collision layers defined. All objects need to be in collision layers."], 3]
    ],
    [
        "#863 Extended Bodies:Multiple Parallel Players, Multiple Extended Object",
        [{file:"test_scripts/863_extended_bodies_multiple_parallel_players_multiple_exten.txt"}, ["line 78 : You're repeating the object PLAYER here multiple times on the RHS. This makes no sense. Don't do that."], 0]
    ],
    [
        "Startloop/endloop errors",
        [{file:"test_scripts/startloop_endloop_errors.txt"}, ["line 83 : Found a STARTLOOP, but I'm already inside a loop? (Puzzlescript can't nest loops, FWIW).", "line 85 : Found an ENDLOOP, but I'm not in a loop?", "line 87 : Yo I found a STARTLOOP without a corresponding ENDLOOP."], 3]
    ],
    [
        "You're talking about TAWETWAT but it's not defined anywhere",
        [{file:"test_scripts/you_re_talking_about_tawetwat_but_it_s_not_defined_anywhere.txt"}, ["line 61 : You're talking about TAWETWAT but it's not defined anywhere."], 1]
    ],
    [
        "broken legend syntax error",
        [{file:"test_scripts/broken_legend_syntax_error.txt"}, ["line 53 : Incorrect format of legend - should be one of \"A = B\", \"A = B or C [ or D ...]\", \"A = B and C [ and D ...]\"."], 1]
    ],
    [
        "Cannot define an aggregate in terms of properties",
        [{file:"test_scripts/cannot_define_an_aggregate_in_terms_of_properties.txt"}, ["line 61 : Cannot define an aggregate (using 'and') in terms of properties (something that uses 'or').", "Trying to define aggregate \"2\" in terms of property \"1\"."], 2]
    ],
    [
        "Cannot define a property in terms of aggregates.",
        [{file:"test_scripts/cannot_define_a_property_in_terms_of_aggregates.txt"}, ["line 61 : Cannot define a property (something defined in terms of 'or') in terms of an aggregate (something that uses 'and'). In this case, you can't define \"2\" in terms of \"1\"."], 1]
    ],
    [
        "Cannot Duplicate Sections",
        [{file:"test_scripts/cannot_duplicate_sections.txt"}, ["line 11 : cannot duplicate sections (you tried to duplicate \"OBJECTS\").", "line 11 : section \"OBJECTS\" must be the first section"], 2]
    ],
    [
        "OBJECTS must be the first section",
        [{file:"test_scripts/objects_must_be_the_first_section.txt"}, ["line 6 : section \"LEGEND\" is out of order, must follow \"OBJECTS\" (or it could be that the section \"OBJECTS\"is just missing totally. You have to include all section headings, even if the section itself is empty).", "line 9 : section \"OBJECTS\" must be the first section", "line 53 : cannot duplicate sections (you tried to duplicate \"LEGEND\")."], 3]
    ],
    [
        "WINCONDITIONS is out of order",
        [{file:"test_scripts/winconditions_is_out_of_order.txt"}, ["line 77 : section \"WINCONDITIONS\" is out of order, must follow \"RULES\" (or it could be that the section \"RULES\"is just missing totally. You have to include all section headings, even if the section itself is empty)."], 1]
    ],
    [
        "name B already in use",
        [{file:"test_scripts/name_b_already_in_use.txt"}, ["line 18 : Name \"B\" already in use.", "line 18 : Object \"B\" has been defined, but not assigned to a layer."], 2]
    ],
    [
        "Unknown junk in spritematrix",
        [{file:"test_scripts/unknown_junk_in_spritematrix.txt"}, ["line 31 : Unknown junk in spritematrix for object WALL.", "line 26 : Sprite graphics must be 5 wide and 5 high exactly."], 1]
    ],
    [
        "In the legend, define new items using the equals symbol",
        [{file:"test_scripts/in_the_legend_define_new_items_using_the_equals_symbol.txt"}, ["line 53 : In the legend, define new items using the equals symbol - declarations must look like \"A = B\", \"A = B or C [ or D ...]\", \"A = B and C [ and D ...]\".", "line 53 : Incorrect format of legend - should be one of \"A = B\", \"A = B or C [ or D ...]\", \"A = B and C [ and D ...]\", but it looks like you have a dangling \"ERROR\"?", "line 90 : Key \".\" not found. Do you need to add it to the legend, or define a new object?", "line 91 : Key \".\" not found. Do you need to add it to the legend, or define a new object?", "line 92 : Key \".\" not found. Do you need to add it to the legend, or define a new object?", "line 93 : Key \".\" not found. Do you need to add it to the legend, or define a new object?"], 6]
    ],
    [
        "Something bad's happening in the LEGEND",
        [{file:"test_scripts/something_bad_s_happening_in_the_legend.txt"}, ["line 57 : Something bad's happening in the LEGEND"], 1]
    ],
    [
        "Hey! You can't go mixing ANDs and ORs in a single legend entry.",
        [{file:"test_scripts/hey_you_can_t_go_mixing_ands_and_ors_in_a_single_legend_entr.txt"}, ["line 57 : Hey! You can't go mixing ANDs and ORs in a single legend entry."], 1]
    ],
    [
        "Expected and 'AND' or an 'OR' here, but got POO instead.",
        [{file:"test_scripts/expected_and_and_or_an_or_here_but_got_poo_instead.txt"}, ["line 57 : Expected and 'AND' or an 'OR' here, but got POO instead. In the legend, define new items using the equals symbol - declarations must look like 'A = B' or 'A = B and C' or 'A = B or C'.", "line 57 : This legend-entry is incorrectly-formatted - it should be one of A = B, A = B or C ( or D ...), A = B and C (and D ...)"], 2]
    ],
    [
        "Was expecting a sound seed here",
        [{file:"test_scripts/was_expecting_a_sound_seed_here.txt"}, ["line 64 : Was expecting a sound seed here (a number like 123123, like you generate by pressing the buttons above the console panel), but found something else."], 1]
    ],
    [
        "Was expecting a sound event here",
        [{file:"test_scripts/was_expecting_a_sound_event_here.txt"}, ["line 64 : Was expecting a sound event (like SFX3, or ENDLEVEL) or an object name, but didn't find either."], 0]
    ],
    [
        "Was expecting a soundverb here",
        [{file:"test_scripts/was_expecting_a_soundverb_here.txt"}, ["line 64 : Was expecting a soundverb here (MOVE, DESTROY, CANTMOVE, or the like), but found something else."], 1]
    ],
    [
        "COLLISIONLAYERS - error detected - unexpected character ",
        [{file:"test_scripts/collisionlayers_error_detected_unexpected_character.txt"}, ["line 70 : error detected - unexpected character '"], 1]
    ],
    [
        "\"aggr\" is an aggregate (defined using \"and\"), and cannot be added to a single layer",
        [{file:"test_scripts/aggr_is_an_aggregate_defined_using_and_and_cannot_be_added_t.txt"}, ["line 72 : \"aggr\" is an aggregate (defined using \"and\"), and cannot be added to a single layer because its constituent objects must be able to coexist.", "line 57 : Object \"CRATE\" has been defined, but not assigned to a layer.", "line 57 : Object \"TARGET\" has been defined, but not assigned to a layer.", "line 58 : Object \"CRATE\" has been defined, but not assigned to a layer.", "line 58 : Object \"TARGET\" has been defined, but not assigned to a layer.", "line 17 : Object \"TARGET\" has been defined, but not assigned to a layer.", "line 41 : Object \"CRATE\" has been defined, but not assigned to a layer."], 7]
    ],
    [
        "Cannot add \"CAT\" to a collision layer; it has not been declared.",
        [{file:"test_scripts/cannot_add_cat_to_a_collision_layer_it_has_not_been_declared.txt"}, ["line 70 : Background must be in a layer by itself.", "line 70 : Cannot add \"CAT\" to a collision layer; it has not been declared."], 2]
    ],
    [
        "Background must be in a layer by itself.",
        [{file:"test_scripts/background_must_be_in_a_layer_by_itself.txt"}, ["line 70 : Background must be in a layer by itself."], 1]
    ],
    [
        "#2 Background must be in a layer by itself.",
        [{file:"test_scripts/2_background_must_be_in_a_layer_by_itself.txt"}, ["line 71 : Background must be in a layer by itself."], 1]
    ],
    [
        "no collision layers defined",
        [{file:"test_scripts/no_collision_layers_defined.txt"}, ["No collision layers defined. All objects need to be in collision layers."], 1]
    ],
    [
        "Identifiers cannot appear outside of square brackets in rules",
        [{file:"test_scripts/identifiers_cannot_appear_outside_of_square_brackets_in_rule.txt"}, ["line 78 : Objects cannot appear outside of square brackets in rules, only directions can.", "line 78 : The start of a rule must consist of some number of directions (possibly 0), before the first bracket, specifying in what directions to look (with no direction specified, it applies in all four directions). It seems you've just entered \"PLAYER\"."], 2]
    ],
    [
        "Name fasd, referred to in a rule, does not exist.",
        [{file:"test_scripts/name_fasd_referred_to_in_a_rule_does_not_exist.txt"}, ["line 78 : Name \"fasd\", referred to in a rule, does not exist.", "line 78 : Error, malformed cell rule - was looking for cell contents, but found \"fasd\". What am I supposed to do with this, eh, please tell me that."], 2]
    ],
    [
        "incorrect format of win condition",
        [{file:"test_scripts/incorrect_format_of_win_condition.txt"}, ["line 84 : incorrect format of win condition."], 1]
    ],
    [
        "Expecting the start of a win condition",
        [{file:"test_scripts/expecting_the_start_of_a_win_condition.txt"}, ["line 84 : Expecting the start of a win condition (\"ALL\",\"SOME\",\"NO\") but got \"FAS'.", "line 84 : Win conditions is badly formatted - needs to look something like \"No Fruit\", \"All Target On Crate\", \"Some Fruit\", \"Some Gold on Chest\", \"No Gold on Chest\", or the like."], 2]
    ],
    [
        "Expecting the word ON but got CRATE",
        [{file:"test_scripts/expecting_the_word_on_but_got_crate.txt"}, ["line 84 : Expecting the word \"ON\" but got \"CRATE\"."], 1]
    ],
    [
        "Key A not found",
        [{file:"test_scripts/key_a_not_found.txt"}, ["line 92 : Key \"A\" not found. Do you need to add it to the legend, or define a new object?", "line 92 : Error, symbol \"a\", used in map, not found."], 2]
    ],
    [
        "Metadata needs a value",
        [{file:"test_scripts/metadata_needs_a_value.txt"}, ["line 3 : MetaData \"homepage\" needs a value.", "line 92 : Key \"A\" not found. Do you need to add it to the legend, or define a new object?", "line 92 : Error, symbol \"a\", used in map, not found."], 3]
    ],
    [
        "a sprite cannot have more than 10 colors",
        [{file:"test_scripts/a_sprite_cannot_have_more_than_10_colors.txt"}, ["line 10 : a sprite cannot have more than 10 colors. Why you would want more than 10 is beyond me."], 1]
    ],
    [
        "Object GREEN has been defined, but not assigned to a layer",
        [{file:"test_scripts/object_green_has_been_defined_but_not_assigned_to_a_layer.txt"}, ["line 50 : Object \"GREEN\" has been defined, but not assigned to a layer."], 1]
    ],
    [
        "ALMOST BLANK test",
        ["========\nOBJECTS\n========\n\n\n=======\nLEGEND\n=======\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\n======\nRULES\n======\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n", ["Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!", "Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!", "No collision layers defined. All objects need to be in collision layers."], 3]
    ],
    [
        "BLANK FILE",
        ["\n", ["Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!", "Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!", "No collision layers defined. All objects need to be in collision layers."], 3]
    ],
    [
        "background cannot be an aggregate",
        [{file:"test_scripts/background_cannot_be_an_aggregate.txt"}, ["line 63 : background cannot be an aggregate (declared with 'and'), it has to be a simple type, or property (declared in terms of others using 'or')."], 1]
    ],
    [
        "Background objects must be on the same layer",
        [{file:"test_scripts/background_objects_must_be_on_the_same_layer.txt"}, ["line 56 : Background objects must be on the same layer"], 1]
    ],
    [
        "Error, symbol A, used in map, not found",
        [{file:"test_scripts/error_symbol_a_used_in_map_not_found.txt"}, ["line 100 : Key \"A\" not found. Do you need to add it to the legend, or define a new object?", "line 100 : Error, symbol \"a\", used in map, not found."], 2]
    ],
    [
        "Symbol is defined using OR, and therefore ambiguous - it cannot be used in a map. Did you mean to define it in terms of AND?",
        [{file:"test_scripts/symbol_is_defined_using_or_and_therefore_ambiguous_it_cannot.txt"}, ["line 101 : Key \"A\" not found. Do you need to add it to the legend, or define a new object?", "line 101 : Error, symbol \"a\" is defined using OR, and therefore ambiguous - it cannot be used in a map. Did you mean to define it in terms of AND?"], 2]
    ],
    [
        "Message too long to fit on screen",
        [{file:"test_scripts/message_too_long_to_fit_on_screen.txt"}, ["line 90 : Message too long to fit on screen."], 0]
    ],
    [
        "A rule has to have an arrow in it part 2.",
        [{file:"test_scripts/a_rule_has_to_have_an_arrow_in_it_part_2.txt"}, ["line 78 : A rule has to have an arrow in it. There's no arrow here! Consider reading up about rules - you're clearly doing something weird", "line 78 : Error, when specifying a rule, the number of matches (square bracketed bits) on the left hand side of the arrow must equal the number on the right"], 1]
    ],
    [
        "The + symbol, for joining a rule with the group of the previous rule, needs a previous rule to be applied to",
        [{file:"test_scripts/the_symbol_for_joining_a_rule_with_the_group_of_the_previous.txt"}, ["line 78 : The \"+\" symbol, for joining a rule with the group of the previous rule, needs a previous rule to be applied to."], 1]
    ],
    [
        "The + symbol, for joining a rule with the group of the previous rule, must be the first symbol on the line",
        [{file:"test_scripts/the_symbol_for_joining_a_rule_with_the_group_of_the_previous_2.txt"}, ["line 79 : The \"+\" symbol, for joining a rule with the group of the previous rule, must be the first symbol on the line"], 1]
    ],
    [
        "two +s appended to same rule",
        [{file:"test_scripts/two_s_appended_to_same_rule.txt"}, ["line 79 : Two \"+\"s (the \"append to previous rule group\" symbol) applied to the same rule."], 1]
    ],
    [
        "A rule-group can only be marked random by the opening rule in the group",
        [{file:"test_scripts/a_rule_group_can_only_be_marked_random_by_the_opening_rule_i_2.txt"}, ["line 79 : A rule-group can only be marked random by the opening rule in the group (aka, a '+' and 'random' can't appear as rule modifiers on the same line). Why? Well, you see \"random\" isn't a property of individual rules, but of whole rule groups. It indicates that a single possible application of some rule from the whole group should be applied at random."], 1]
    ],
    [
        "You cannot use relative directions [^v<>] to indicate in which direction[s] a rule applies",
        [{file:"test_scripts/you_cannot_use_relative_directions_v_to_indicate_in_which_di.txt"}, ["line 78 : Name \"^\", referred to in a rule, does not exist.", "line 78 : You cannot use relative directions (\"^v<>\") to indicate in which direction(s) a rule applies. Use absolute directions indicators (Up, Down, Left, Right, Horizontal, or Vertical, for instance), or, if you want the rule to apply in all four directions, do not specify directions"], 2]
    ],
    [
        "The start of a rule must consist of some number of directions",
        [{file:"test_scripts/the_start_of_a_rule_must_consist_of_some_number_of_direction.txt"}, ["line 78 : Name \"'\", referred to in a rule, does not exist.", "line 78 : The start of a rule must consist of some number of directions (possibly 0), before the first bracket, specifying in what directions to look (with no direction specified, it applies in all four directions). It seems you've just entered \"'\"."], 2]
    ],
    [
        "Error, malformed cell rule",
        [{file:"test_scripts/error_malformed_cell_rule.txt"}, ["line 78 : Multiple opening brackets without closing brackets. Something fishy here. Every '[' has to be closed by a ']', and you can't nest them.", "line 78 : Error, malformed cell rule - encountered a \"[\"\" before previous bracket was closed"], 1]
    ],
    [
        "Invalid syntax. Directions should be placed at the start of a rule",
        [{file:"test_scripts/invalid_syntax_directions_should_be_placed_at_the_start_of_a.txt"}, ["line 78 : Invalid syntax. Directions should be placed at the start of a rule."], 0]
    ],
    [
        "Movements cannot appear in late rules.",
        [{file:"test_scripts/movements_cannot_appear_in_late_rules.txt"}, ["line 78 : Movements cannot appear in late rules."], 1]
    ],
    [
        "| should only be used inside cell rows",
        [{file:"test_scripts/should_only_be_used_inside_cell_rows.txt"}, ["line 78 : Janky syntax. \"|\" should only be used inside cell rows (the square brackety bits)."], 0]
    ],
    [
        "In a rule, if you specify a movement, it has to act on an object",
        [{file:"test_scripts/in_a_rule_if_you_specify_a_movement_it_has_to_act_on_an_obje.txt"}, ["line 78 : In a rule, if you specify a movement, it has to act on an object.", "line 78 : In a rule, each pattern to match on the left must have a corresponding pattern on the right of equal length (number of cells)."], 2]
    ],
    [
        "#2: In a rule, if you specify a movement, it has to act on an object",
        [{file:"test_scripts/2_in_a_rule_if_you_specify_a_movement_it_has_to_act_on_an_ob.txt"}, ["line 78 : In a rule, if you specify a movement, it has to act on an object.", "line 78 : Error, malformed cell rule - encountered a \"[\"\" before previous bracket was closed", "line 78 : In a rule, each pattern to match on the left must have a corresponding pattern on the right of equal length (number of cells)."], 3]
    ],
    [
        "Multiple closing brackets without corresponding opening brackets",
        [{file:"test_scripts/multiple_closing_brackets_without_corresponding_opening_brac.txt"}, ["line 78 : Multiple closing brackets without corresponding opening brackets. Something fishy here. Every '[' has to be closed by a ']', and you can't nest them."], 0]
    ],
    [
        "There's no point in putting an ellipsis at the very start or the end of a rule",
        [{file:"test_scripts/there_s_no_point_in_putting_an_ellipsis_at_the_very_start_or.txt"}, ["line 78 : There's no point in putting an ellipsis at the very start or the end of a rule", "line 78 : An ellipsis on the left must be matched by one in the corresponding place on the right."], 2]
    ],
    [
        "An ellipsis on the right must be matched by one in the corresponding place on the left.",
        [{file:"test_scripts/an_ellipsis_on_the_right_must_be_matched_by_one_in_the_corre.txt"}, ["line 78 : An ellipsis on the right must be matched by one in the corresponding place on the left.", "line 78 : There's no point in putting an ellipsis at the very start or the end of a rule", "line 78 : An ellipsis on the left must be matched by one in the corresponding place on the right."], 3]
    ],
    [
        "You can't have anything in with an ellipsis. Sorry.",
        [{file:"test_scripts/you_can_t_have_anything_in_with_an_ellipsis_sorry_2.txt"}, ["line 78 : You can't have anything in with an ellipsis. Sorry."], 1]
    ],
    [
        "RANDOM cannot be matched on the left-hand side",
        [{file:"test_scripts/random_cannot_be_matched_on_the_left_hand_side.txt"}, ["line 78 : RANDOM cannot be matched on the left-hand side, it can only appear on the right"], 1]
    ],
    [
        "An ellipsis on the right must be matched by one in the corresponding place on the left.",
        [{file:"test_scripts/an_ellipsis_on_the_right_must_be_matched_by_one_in_the_corre_2.txt"}, ["line 78 : An ellipsis on the right must be matched by one in the corresponding place on the left."], 1]
    ],
    [
        "123123 You can't have anything in with an ellipsis. Sorry.",
        [{file:"test_scripts/123123_you_can_t_have_anything_in_with_an_ellipsis_sorry.txt"}, ["line 78 : You can't have anything in with an ellipsis. Sorry."], 1]
    ],
    [
        "This rule may try to spawn a WALL with random, but also requires a CRATE be here, which is on the same layer - they shouldn't be able to coexist!",
        [{file:"test_scripts/this_rule_may_try_to_spawn_a_wall_with_random_but_also_requi.txt"}, ["line 79 : This rule may try to spawn a WALL with random, but also requires a CRATE be here, which is on the same layer - they shouldn't be able to coexist!", "line 1 : Game title is too long to fit on screen; truncating to fit."], 0]
    ],
    [
        "WALL and CRATE can never overlap. This rule requires that to happen, so it impossible it will ever trigger.",
        [{file:"test_scripts/wall_and_crate_can_never_overlap_this_rule_requires_that_to.txt"}, ["line 81 : WALL and CRATE can never overlap. This rule requires that to happen, so it impossible it will ever trigger."], 1]
    ],
    [
        "There can't be more than 30 rigid groups",
        [{file:"test_scripts/there_can_t_be_more_than_30_rigid_groups.txt"}, ["line 141 : There can't be more than 30 rigid groups (rule groups containing rigid members)."], 1]
    ],
    [
        "error, didn't find any object called player",
        [{file:"test_scripts/error_didn_t_find_any_object_called_player.txt"}, ["Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!", "Error, didn't find any object called player, either in the objects section, or the legends section."], 2]
    ],
    [
        "I wasn't expecting anything after the sound declaration",
        [{file:"test_scripts/i_wasn_t_expecting_anything_after_the_sound_declaration.txt"}, ["line 67 : I wasn't expecting anything after the sound declaration 36772507 on this line, so I don't know what to do with \"ASDASD\" here."], 1]
    ],
    [
        "invalid text_color / background_color",
        [{file:"test_scripts/invalid_text_color_background_color.txt"}, ["line 4 : text_color in incorrect format - found 21r3r23r2, but I expect a color name (like 'pink') or hex-formatted color (like '#1412FA'). Defaulting to white.", "line 5 : background_color in incorrect format - found 21r3r23r2, but I expect a color name (like 'pink') or hex-formatted color (like '#1412FA'). Defaulting to black."], 2]
    ],
    [
        "Game title is too long to fit on screen; truncating to fit.",
        [{file:"test_scripts/game_title_is_too_long_to_fit_on_screen_truncating_to_fit_2.txt"}, ["line 1 : Game title is too long to fit on screen; truncating to fit."], 0]
    ],
    [
        "Author name is too long to fit on screen",
        [{file:"test_scripts/author_name_is_too_long_to_fit_on_screen.txt"}, ["line 1 : Author name is too long to fit on screen; truncating to fit."], 0]
    ],
    [
        "In this rule you're asking me to spawn a random CRATE for you, but that's already a concrete single object.",
        [{file:"test_scripts/in_this_rule_you_re_asking_me_to_spawn_a_random_crate_for_yo.txt"}, ["line 78 : In this rule you're asking me to spawn a random CRATE for you, but that's already a concrete single object. You wanna be using random with properties (things defined in terms of OR in the legend) so there's some things to select between."], 0]
    ],
    [
        "putting random before an aggregate has something funny, lol",
        [{file:"test_scripts/putting_random_before_an_aggregate_has_something_funny_lol.txt"}, ["line 82 : In this rule you're asking me to spawn a random CRATE for you, but that's already a concrete single object. You wanna be using random with properties (things defined in terms of OR in the legend) so there's some things to select between.", "line 82 : In this rule you're asking me to spawn a random TARGET for you, but that's already a concrete single object. You wanna be using random with properties (things defined in terms of OR in the legend) so there's some things to select between."], 0]
    ],
    [
        "MetaData DEBUG doesn't take any parameters, but you went and gave it",
        [{file:"test_scripts/metadata_debug_doesn_t_take_any_parameters_but_you_went_and.txt"}, ["line 2 : MetaData DEBUG doesn't take any parameters, but you went and gave it \"baff\"."], 0]
    ],
    [
        "metadata parameter errors Part 2",
        [{file:"test_scripts/metadata_parameter_errors_part_2.txt"}, ["line 5 : MetaData DEBUG doesn't take any parameters, but you went and gave it \"asd\".", "line 6 : MetaData NOACTION doesn't take any parameters, but you went and gave it \"asdf\".", "line 7 : MetaData NOREPEAT_ACTION doesn't take any parameters, but you went and gave it \"asdf\".", "line 8 : MetaData NOUNDO doesn't take any parameters, but you went and gave it \"4324\".", "line 9 : MetaData NORESTART doesn't take any parameters, but you went and gave it \"nsfgn\".", "line 10 : MetaData REQUIRE_PLAYER_MOVEMENT doesn't take any parameters, but you went and gave it \"123\".", "line 11 : MetaData RUN_RULES_ON_LEVEL_START doesn't take any parameters, but you went and gave it \"123\".", "line 12 : MetaData SCANLINE doesn't take any parameters, but you went and gave it \"4234\".", "line 13 : MetaData THROTTLE_MOVEMENT doesn't take any parameters, but you went and gave it \"434\".", "line 14 : MetaData VERBOSE_LOGGING doesn't take any parameters, but you went and gave it \"232\"."], 0]
    ],
    [
        "Error in win condition",
        [{file:"test_scripts/error_in_win_condition.txt"}, ["line 84 : Error in win condition: I don't know what to do with CRATE."], 1]
    ],
    [
        "late+rigid rules",
        [{file:"test_scripts/late_rigid_rules.txt"}, ["line 78 : Late rules cannot be marked as rigid (rigid rules are all about dealing with the consequences of unresolvable movements, and late rules can't even have movements)."], 1]
    ],
    [
        "Duplicate prelude declarations",
        [{file:"test_scripts/duplicate_prelude_declarations.txt"}, ["line 4 : You've already defined a HOMEPAGE in the prelude on line 3."], 0]
    ],
    [
        "flickscreen/zoomscreen formatting error #1",
        [{file:"test_scripts/flickscreen_zoomscreen_formatting_error_1.txt"}, ["line 6 : Couldn't understand the dimensions given to me (you gave \"2ax10\") - should be of the form AxB.", "line 5 : Couldn't understand the dimensions given to me (you gave \"2ax10\") - should be of the form AxB."], 0]
    ],
    [
        "flickscreen/zoomscreen formatting error #2",
        [{file:"test_scripts/flickscreen_zoomscreen_formatting_error_2.txt"}, ["line 6 : Dimensions must be of the form AxB.", "line 5 : Dimensions must be of the form AxB."], 0]
    ],
    [
        "Dimensions should be positive",
        [{file:"test_scripts/dimensions_should_be_positive.txt"}, ["line 5 : The dimensions given to me (you gave \"-1x-3\") are baad - they should be > 0."], 0]
    ],
    [
        "Dimensions should be positive and integral",
        [{file:"test_scripts/dimensions_should_be_positive_and_integral.txt"}, ["line 6 : Wasn't able to make sense of \"1.4\" as a (whole number) dimension.", "line 6 : The dimension given to me (you gave \"-4\") is baad - it should be greater than 0.", "line 6 : Couldn't understand the dimensions given to me (you gave \"1.4x-4\") - should be of the form AxB.", "line 5 : The dimension given to me (you gave \"0\") is baad - it should be greater than 0.", "line 5 : Wasn't able to make sense of \"4.23\" as a (whole number) dimension.", "line 5 : Couldn't understand the dimensions given to me (you gave \"0x4.23\") - should be of the form AxB."], 0]
    ],
    [
        "Match 3 Block Push",
        [{file:"test_scripts/match_3_block_push.txt"}, ["line 3 : Error: you can't embed comments in metadata values. Anything after the comment will be ignored."], 0]
    ],
    [
        "Mixing late and non-late in rulegroups",
        [{file:"test_scripts/mixing_late_and_non_late_in_rulegroups.txt"}, ["line 82 : Oh gosh you can mix late and non-late rules in a rule-group if you really want to, but gosh why would you want to do that? What do you expect to accomplish?", "line 83 : Oh gosh you can mix late and non-late rules in a rule-group if you really want to, but gosh why would you want to do that? What do you expect to accomplish?", "line 87 : Oh gosh you can mix late and non-late rules in a rule-group if you really want to, but gosh why would you want to do that? What do you expect to accomplish?"], 0]
    ],
    [
        "loop points in the middle of rule-groups",
        [{file:"test_scripts/loop_points_in_the_middle_of_rule_groups.txt"}, ["line 82 : Found a loop point in the middle of a rule. You probably don't want to do this, right?"], 1]
    ],
    [
        "can declare the same object to be on a layer multiple times #770",
        [{file:"test_scripts/can_declare_the_same_object_to_be_on_a_layer_multiple_times.txt"}, ["line 49 : Object \"PLAYER\" included explicitly multiple times in the same layer. Don't do that innit."], 0]
    ],
    [
        "Double-Ellipsis error message neighbouring ellipses",
        [{file:"test_scripts/double_ellipsis_error_message_neighbouring_ellipses.txt"}, ["line 81 : Why would you go and have two ellipses in a row like that? It's exactly the same as just having a single ellipsis, right?"], 0]
    ],
    [
        "Simple Block Pushing Game",
        [{file:"test_scripts/simple_block_pushing_game.txt"}, ["line 78 : You can't use more than two ellipses in a single cell match pattern."], 1]
    ],
    [
        "Warning for duplicate objects on layer",
        [{file:"test_scripts/warning_for_duplicate_objects_on_layer.txt"}, ["line 72 : Object \"WALL\" included explicitly multiple times in the same layer. Don't do that innit."], 0]
    ],
    [
        "No warning for duplicate objects on layer",
        [{file:"test_scripts/no_warning_for_duplicate_objects_on_layer.txt"}, [], 0]
    ],
    [
        "Warning for duplicate properties on layer",
        [{file:"test_scripts/warning_for_duplicate_properties_on_layer.txt"}, ["line 72 : Object \"COW\" included explicitly multiple times in the same layer. Don't do that innit."], 0]
    ],
    [
        "Ambiguous properties throw an exception #929",
        ["OBJECTS\n\nBackground\nblack\n\nPlayer\nblue\n\nA\ngreen\n\nB\nred\n\nLEGEND\nC = A or B\n\nSOUNDS\nCOLLISIONLAYERS\n\nA \nB\n\nRULES\nright [ C | C | ] -> [ C | C | C ]\n\nWINCONDITIONS\nLEVELS\n", ["line 25 : This rule has a property on the right-hand side, \"C\", that can't be inferred from the left-hand side. (either for every property on the right there has to be a corresponding one on the left in the same cell, OR, if there's a single occurrence of a particular property name on the left, all properties of the same name on the right are assumed to be the same).", "line 3 : Object \"BACKGROUND\" has been defined, but not assigned to a layer.", "line 6 : Object \"PLAYER\" has been defined, but not assigned to a layer.", "No levels found. Add some levels!"], 4]
    ],
    [
        "color palette index error",
        [{file:"test_scripts/color_palette_index_error.txt"}, ["line 0 : Palette \"41\" not found, defaulting to arnecolors."], 1]
    ],
    [
        "color palette name error",
        [{file:"test_scripts/color_palette_name_error.txt"}, ["line 0 : Palette \"aerawerawer\" not found, defaulting to arnecolors."], 1]
    ],
    [
        "you need to have some objects defined",
        ["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\n========\nOBJECTS\n========\n\n=======\nLEGEND\n=======\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nplayer\n\n======\nRULES\n======\n \n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n", ["line 21 : Cannot add \"PLAYER\" to a collision layer; it has not been declared.", "Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!", "Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!", "You need to have some objects defined", "Seriously, you have to define something to be the background.", "Error, didn't find any object called player, either in the objects section, or the legends section.", "No levels found. Add some levels!"], 7]
    ],
    [
        "test to cover rightBracketToRightOf",
        [{file:"test_scripts/test_to_cover_rightbrackettorightof.txt"}, ["line 81 : Commands should only appear at the end of rules, not in or before the pattern-detection/-replacement sections.", "line 81 : Sound effect \"sfx0\" not defined.", "line 81 : Multiple closing brackets without corresponding opening brackets. Something fishy here. Every '[' has to be closed by a ']', and you can't nest them."], 0]
    ],
    [
        "Error, an item can only have one direction/action at a time",
        [{file:"test_scripts/error_an_item_can_only_have_one_direction_action_at_a_time.txt"}, ["line 78 : Error, an item can only have one direction/action at a time, but you're looking for several at once!"], 1]
    ],
    [
        "title Error, you can only use \"->\" once in a rule",
        [{file:"test_scripts/title_error_you_can_only_use_once_in_a_rule.txt"}, ["line 78 : Error, you can only use \"->\" once in a rule; it's used to separate before and after states."], 1]
    ],
    [
        "Object names should only be used within cells",
        [{file:"test_scripts/object_names_should_only_be_used_within_cells.txt"}, ["line 78 : Invalid token PLAYER. Object names should only be used within cells (square brackets)."], 0]
    ],
    [
        "ellipses should only be used within cells",
        [{file:"test_scripts/ellipses_should_only_be_used_within_cells.txt"}, ["line 78 : Invalid syntax, ellipses should only be used within cells (square brackets)."], 0]
    ],
    [
        "This rule refers to nothing",
        [{file:"test_scripts/this_rule_refers_to_nothing.txt"}, ["line 78 : The start of a rule must consist of some number of directions (possibly 0), before the first bracket, specifying in what directions to look (with no direction specified, it applies in all four directions). It seems you've just entered \"->\".", "line 78 : This rule refers to nothing. What the heck? :O"], 2]
    ],
    [
        "You cannot use 'no' to exclude the aggregate object",
        [{file:"test_scripts/you_cannot_use_no_to_exclude_the_aggregate_object.txt"}, ["line 78 : You cannot use 'no' to exclude the aggregate object @ (defined using 'AND'), only regular objects, or properties (objects defined using 'OR'). If you want to do this, you'll have to write it out yourself the long way."], 1]
    ],
    [
        "Unwelcome term error in second slot",
        [{file:"test_scripts/unwelcome_term_error_in_second_slot.txt"}, ["line 84 : Error in win condition: \"UP\" is not a valid object name.", "line 84 : Unwelcome term \"up\" found in win condition. I don't know what I'm supposed to do with this."], 2]
    ],
    [
        "cannot assign sound events to aggregate objects",
        [{file:"test_scripts/cannot_assign_sound_events_to_aggregate_objects.txt"}, ["line 64 : Expecting sfx data, instead found \"create\".", "line 64 : cannot assign sound events to aggregate objects (declared with \"and\"), only to regular objects, or properties, things defined in terms of \"or\" (\"at\")."], 2]
    ],
    [
        "Hey! You can't go mixing ANDs and ORs #2",
        [{file:"test_scripts/hey_you_can_t_go_mixing_ands_and_ors_2.txt"}, ["line 57 : Hey! You can't go mixing ANDs and ORs in a single legend entry.", "line 57 : You're repeating the object CRATE here multiple times on the RHS. This makes no sense. Don't do that.", "line 93 : Key \"@\" not found. Do you need to add it to the legend, or define a new object?", "line 101 : Key \"@\" not found. Do you need to add it to the legend, or define a new object?", "line 102 : Key \"@\" not found. Do you need to add it to the legend, or define a new object?", "line 93 : Error, symbol \"@\", used in map, not found.", "line 101 : Error, symbol \"@\", used in map, not found."], 6]
    ],
    [
        "parser sound error highlighting fallback stuff",
        [{file:"test_scripts/parser_sound_error_highlighting_fallback_stuff.txt"}, ["line 64 : Ah I was expecting direction or a sound seed here after MOVE, but I don't know what to make of \"MOVE\"."], 1]
    ],
    [
        "#1007 Unwelcome term \"84\" found in win condition",
        [{file:"test_scripts/1007_unwelcome_term_84_found_in_win_condition.txt"}, ["line 84 : Win conditions is badly formatted - needs to look something like \"No Fruit\", \"All Target On Crate\", \"Some Fruit\", \"Some Gold on Chest\", \"No Gold on Chest\", or the like."], 1]
    ],
    [
        "can strew dots around the prelude lol #1008",
        [{file:"test_scripts/can_strew_dots_around_the_prelude_lol_1008.txt"}, ["line 1 : Unrecognised stuff \".title simple block pushing game.. .\" in the prelude.", "line 2 : Unrecognised stuff \".author david skinner..\" in the prelude.", "line 5 : Unrecognised stuff \"..debug\" in the prelude.", "line 6 : Unrecognised stuff \"..verbose_logging\" in the prelude.", "line 7 : Unrecognised stuff \".\" in the prelude.", "line 8 : Unrecognised stuff \"..\" in the prelude."], 6]
    ],
    [
        "tidy up ===== parsing #1033",
        [{file:"test_scripts/tidy_up_parsing_1033.txt"}, [], 0]
    ],
    [
        "superfluous \"no x\" rule checker",
        [{file:"test_scripts/superfluous_no_x_rule_checker.txt"}, ["line 97 : You have specified that there should be NO WALL but there is also a requirement that PLAYER be there, which collectively occupies the same layers (Layers 3), so you can leave this out.", "line 101 : You have specified that there should be NO CRATE2_OR_4 but there is also a requirement that PLAYER be there, which collectively occupies the same layers (Layers 3), so you can leave this out.", "line 107 : You have specified that there should be NO CRATE5_OR_6 but there is also a requirement that PLAYER, CRATE7 be there, which collectively occupies the same layers (Layers 3, 4), so you can leave this out."], 0]
    ],
    [
        "#1067 [ player no wall ] test too many warnings test",
        [{file:"test_scripts/1067_player_no_wall_test_too_many_warnings_test.txt"}, ["line 158 : You have specified that there should be NO CRATE but there is also a requirement that PLAYER be there, which collectively occupies the same layers (Layers 3), so you can leave this out.", "line 164 : You have specified that there should be NO WALL_OR_CRATE but there is also a requirement that PLAYER be there, which collectively occupies the same layers (Layers 3), so you can leave this out.", "line 176 : You have specified that there should be NO PLAYER but there is also a requirement that WALL_OR_CRATE be there, which collectively occupies the same layers (Layers 3), so you can leave this out.", "line 212 : You have specified that there should be NO WALL but there is also a requirement that PLAYER_AND_TARGET, PLAYER_AND_TARGET be there, which collectively occupies the same layers (Layers 2, 3), so you can leave this out.", "line 214 : You have specified that there should be NO TARGET but there is also a requirement that PLAYER_AND_TARGET, PLAYER_AND_TARGET be there, which collectively occupies the same layers (Layers 2, 3), so you can leave this out.", "line 218 : You have specified that there should be NO WALL_OR_CRATE but there is also a requirement that PLAYER_AND_TARGET, PLAYER_AND_TARGET be there, which collectively occupies the same layers (Layers 2, 3), so you can leave this out.", "line 226 : You have specified that there should be NO CRATE_OR_TARGET but there is also a requirement that PLAYER_AND_TARGET, PLAYER_AND_TARGET be there, which collectively occupies the same layers (Layers 2, 3), so you can leave this out.", "line 240 : You have specified that there should be NO WALL but there is also a requirement that PLAYER_OR_CRATE be there, which collectively occupies the same layers (Layers 3), so you can leave this out.", "line 270 : You have specified that there should be NO WALL but there is also a requirement that PLAYER be there, which collectively occupies the same layers (Layers 3), so you can leave this out.", "line 271 : You have specified that there should be NO L4A but there is also a requirement that L4B be there, which collectively occupies the same layers (Layers 4), so you can leave this out.", "line 277 : You have specified that there should be NO L4C but there is also a requirement that L4A_OR_L4B be there, which collectively occupies the same layers (Layers 4), so you can leave this out.", "line 287 : You have specified that there should be NO L4A_OR_L4B but there is also a requirement that L4C be there, which collectively occupies the same layers (Layers 4), so you can leave this out.", "line 295 : You have specified that there should be NO L4A_OR_L5A but there is also a requirement that L4C, L5C be there, which collectively occupies the same layers (Layers 4, 5), so you can leave this out.", "line 310 : You have specified that there should be NO L4A_OR_L4B but there is also a requirement that L4C be there, which collectively occupies the same layers (Layers 4), so you can leave this out.", "line 318 : You have specified that there should be NO L4A_OR_L5A but there is also a requirement that L4B, L5B be there, which collectively occupies the same layers (Layers 4, 5), so you can leave this out.", "line 160 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 166 : This rule has something amounting to \"WALL NO WALL\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 172 : This rule has something amounting to \"CRATE NO CRATE\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 196 : This rule has something amounting to \"TARGET NO TARGET\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 202 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 208 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 220 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 224 : This rule has something amounting to \"TARGET2 NO TARGET2\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 255 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 282 : This rule has something amounting to \"L4A NO L4A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 282 : This rule has something amounting to \"L5A NO L5A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 292 : This rule has something amounting to \"L4A NO L4A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 316 : This rule has something amounting to \"L4A NO L4A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 1",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_1.txt"}, ["line 155 : You have specified that there should be NO L4A_OR_L4B but there is also a requirement that L4A be here. This is a mistake right?"], 1]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 2",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_2.txt"}, ["line 156 : You have specified that there should be NO L4A_OR_L5A but there is also a requirement that L4A be here. This is a mistake right?"], 1]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 3",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_3.txt"}, ["line 182 : You have specified that there should be NO CRATE but there is also a requirement that PLAYER be there, which collectively occupies the same layers (Layers 3), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 4",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_4.txt"}, ["line 184 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 5",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_5.txt"}, ["line 188 : You have specified that there should be NO WALL_OR_CRATE but there is also a requirement that PLAYER be there, which collectively occupies the same layers (Layers 3), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 5",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_5_2.txt"}, ["line 190 : This rule has something amounting to \"WALL NO WALL\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 7",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_7.txt"}, ["line 196 : This rule has something amounting to \"CRATE NO CRATE\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 8",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_8.txt"}, ["line 200 : You have specified that there should be NO PLAYER but there is also a requirement that WALL_OR_CRATE be there, which collectively occupies the same layers (Layers 3), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 9",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_9.txt"}, ["line 220 : This rule has something amounting to \"TARGET NO TARGET\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 10",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_10.txt"}, ["line 226 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 11",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_11.txt"}, ["line 232 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 12",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_12.txt"}, ["line 236 : You have specified that there should be NO WALL but there is also a requirement that PLAYER_AND_TARGET, PLAYER_AND_TARGET be there, which collectively occupies the same layers (Layers 2, 3), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 13",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_13.txt"}, ["line 238 : You have specified that there should be NO TARGET but there is also a requirement that PLAYER_AND_TARGET, PLAYER_AND_TARGET be there, which collectively occupies the same layers (Layers 2, 3), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 14",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_14.txt"}, ["line 242 : You have specified that there should be NO WALL_OR_CRATE but there is also a requirement that PLAYER_AND_TARGET, PLAYER_AND_TARGET be there, which collectively occupies the same layers (Layers 2, 3), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 15",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_15.txt"}, ["line 244 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 16",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_16.txt"}, ["line 248 : This rule has something amounting to \"TARGET2 NO TARGET2\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 17",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_17.txt"}, ["line 250 : You have specified that there should be NO CRATE_OR_TARGET but there is also a requirement that PLAYER_AND_TARGET, PLAYER_AND_TARGET be there, which collectively occupies the same layers (Layers 2, 3), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 18",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_18.txt"}, ["line 264 : You have specified that there should be NO WALL but there is also a requirement that PLAYER_OR_CRATE be there, which collectively occupies the same layers (Layers 3), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 19",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_19.txt"}, ["line 280 : You have specified that there should be NO L4A_OR_L4B but there is also a requirement that L4A be here. This is a mistake right?"], 1]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 20",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_20.txt"}, ["line 286 : You have specified that there should be NO WALL but there is also a requirement that PLAYER be there, which collectively occupies the same layers (Layers 3), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 21",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_21.txt"}, ["line 287 : You have specified that there should be NO L4A but there is also a requirement that L4B be there, which collectively occupies the same layers (Layers 4), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 22",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_22.txt"}, ["line 293 : You have specified that there should be NO L4C but there is also a requirement that L4A_OR_L4B be there, which collectively occupies the same layers (Layers 4), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 23",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_23.txt"}, ["line 298 : This rule has something amounting to \"L4A NO L4A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.", "line 298 : This rule has something amounting to \"L5A NO L5A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 24",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_24.txt"}, ["line 303 : You have specified that there should be NO L4A_OR_L4B but there is also a requirement that L4C be there, which collectively occupies the same layers (Layers 4), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 25",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_25.txt"}, ["line 308 : This rule has something amounting to \"L4A NO L4A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 26",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_26.txt"}, ["line 311 : You have specified that there should be NO L4A_OR_L5A but there is also a requirement that L4C, L5C be there, which collectively occupies the same layers (Layers 4, 5), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 27",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_27.txt"}, ["line 326 : You have specified that there should be NO L4A_OR_L4B but there is also a requirement that L4C be there, which collectively occupies the same layers (Layers 4), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 28",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_28.txt"}, ["line 332 : This rule has something amounting to \"L4A NO L4A\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 29",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_29.txt"}, ["line 334 : You have specified that there should be NO L4A_OR_L5A but there is also a requirement that L4B, L5B be there, which collectively occupies the same layers (Layers 4, 5), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 30",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_30.txt"}, ["line 334 : You have specified that there should be NO L4A_OR_L5A but there is also a requirement that L4B, L5B be there, which collectively occupies the same layers (Layers 4, 5), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 31",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_31.txt"}, ["line 341 : You have specified that there should be NO L4A_OR_L4B but there is also a requirement that 14C_AND_15C, 14C_AND_15C be there, which collectively occupies the same layers (Layers 4, 5), so you can leave this out."], 0]
    ],

    [
        "#1067 [ player no wall ] message test suite, No. 32",
        [{file:"test_scripts/1067_player_no_wall_message_test_suite_no_32.txt"}, ["line 343 : You have specified that there should be NO L4A_OR_L5A but there is also a requirement that L4A, L5A be here. This is a mistake right?"], 1]
    ],
    [
        "bad error if parenthetical inside rule #1128",
        [{file:"test_scripts/bad_error_if_parenthetical_inside_rule_1128.txt"},["line 78 : You can't have comments inside rules, sorry. In PuzzleScript, '(' starts a comment, so everything after the first '(' on this line was ignored before your rule was parsed."],1]
    ],
    [
        "fixes #1163 (Vertical tab can cause compiler to hang)",
        ["RULES\n¸\u000b\n",["line 1 : section \"RULES\" is out of order, must follow \"COLLISIONLAYERS\" (or it could be that the section \"COLLISIONLAYERS\"is just missing totally. You have to include all section headings, even if the section itself is empty).","line 2 : Name \"¸\", referred to in a rule, does not exist.","Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!","Error, didn't find any object called background, either in the objects section, or the legends section. There must be a background!","No collision layers defined. All objects need to be in collision layers."],5]
    ],
    [
        "Property inferred overwrite regression in puzzlescript-labs (Thing ObjA)",
        ["title Property inferred overwrite\n========\nOBJECTS\n========\nBackground\nblack\nPlayer\nwhite\nObjA\nred\nObjB\nblue\n========\nLEGEND\n========\n. = Background\nP = Player\nx = ObjA\ny = ObjB\nThing = ObjA or ObjB\n========\nSOUNDS\n========\n================\nCOLLISIONLAYERS\n================\nBackground\nPlayer\nObjA\nObjB\n=====\nRULES\n=====\n[ Thing ] -> [ Thing ObjA ]\n=============\nWINCONDITIONS\n=============\nSome Player\n======\nLEVELS\n======\nPy\n\n",["line 34 : Rule matches object types that can't overlap: \"OBJA\" and \"OBJA\"."],1]
    ],
    [
        "Property inferred overwrite regression in puzzlescript-labs (ObjA Thing)",
        ["title Property inferred overwrite\n========\nOBJECTS\n========\nBackground\nblack\nPlayer\nwhite\nObjA\nred\nObjB\nblue\n========\nLEGEND\n========\n. = Background\nP = Player\nx = ObjA\ny = ObjB\nThing = ObjA or ObjB\n========\nSOUNDS\n========\n================\nCOLLISIONLAYERS\n================\nBackground\nPlayer\nObjA\nObjB\n=====\nRULES\n=====\n[ Thing ] -> [ ObjA Thing ]\n=============\nWINCONDITIONS\n=============\nSome Player\n======\nLEVELS\n======\nPy\n\n",["line 34 : Rule matches object types that can't overlap: \"OBJA\" and \"OBJA\"."],1]
    ],
    [
        "#1169 [ ] -> [ no Player no Player ] should warn not error",
        ["title #1169 [ ] -> [ no Player no Player ] should warn not error\nauthor David Skinner\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\n[ ] -> [ no Player no Player ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n\n",["line 36 : I notice that you have NO PLAYER more than once in a single cell. That's redundant innit."],0]
    ],
    [
        "#1164 type error on compilation (with repeated words)",
        [{file:"test_scripts/1164_type_error_on_compilation_with_repeated_words.txt"},["line 81 : You cannot specify the same object more than once in a single cell (in this case PLAYER occurs multiple times).","line 82 : You cannot specify the same object more than once in a single cell (in this case PLAYER occurs multiple times).","line 81 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.","line 82 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."],2]
    ],
    [
        "#1136 multiple 'X no X' cells in one rule can lead to unexpected rule deletion",
        [{file:"test_scripts/1136_multiple_x_no_x_cells_in_one_rule_can_lead_to_unexpecte.txt"},["line 82 : You cannot specify the same object more than once in a single cell (in this case PLAYER occurs multiple times).","line 82 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."],1]
    ],

    [
        "#1136 multiple 'X no X' cells in one rule can lead to unexpected rule deletion - duplicated second rule",
        [{file:"test_scripts/1136_multiple_x_no_x_cells_in_one_rule_can_lead_to_unexpecte_2.txt"},["line 82 : You cannot specify the same object more than once in a single cell (in this case PLAYER occurs multiple times).","line 83 : You cannot specify the same object more than once in a single cell (in this case PLAYER occurs multiple times).","line 82 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation.","line 83 : This rule has something amounting to \"PLAYER NO PLAYER\" on the left-hand-side,which can never match, so the rule is getting removed during compilation."],2]
    ],
    [
        "unclosed '[' in a rule should error instead of crashing (rewriteUpLeftRules reverse)",
        ["title unclosed bracket in rule\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = Crate\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate\n\n======\nRULES\n======\n\n[> Player][stationary Crate]->[> Player][> Crate\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 40 : Error, when specifying a rule, the number of matches (square bracketed bits) on the left hand side of the arrow must equal the number on the right"],0]
    ],
    [
        "RHS longer than LHS with 'no' should error instead of crashing (lhs_cell.length)",
        ["title rhs longer than lhs\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = Crate\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate\n\n======\nRULES\n======\n\n[ > Player ] -> [ > Player | no Crate ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 40 : In a rule, each pattern to match on the left must have a corresponding pattern on the right of equal length (number of cells)."],1]
    ],
    [
        "no aggregate on the RHS should error instead of crashing (anyBitsInCommon)",
        ["title no aggregate on RHS\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nTarget\nblue\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nO = Target\nBoth = Player and Target\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer\n\n======\nRULES\n======\n\n[ Player ] -> [ Player no Both ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 42 : You cannot use 'no' to exclude the aggregate object BOTH (defined using 'AND'), only regular objects, or properties (objects defined using 'OR'). If you want to do this, you'll have to write it out yourself the long way."],1]
    ],
    [
        "property listing an aggregate and used through a synonym should error instead of crashing (layer)",
        ["title property contains aggregate\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nTarget\nblue\n\nCrate\norange\n\nWall\ngray\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nO = Target\n* = Crate\n# = Wall\nBoth = Player and Target\nStuff = Crate or Wall or Both\nAlias = Stuff\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Crate, Wall\n\n======\nRULES\n======\n\n[ Alias ] -> [ Alias ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 33 : Cannot define a property (something defined in terms of 'or') in terms of an aggregate (something that uses 'and'). In this case, you can't define \"STUFF\" in terms of \"BOTH\"."],1]
    ],
    [
        "synonym of janky aggregate",
        ["title synonym of janky aggregate\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nTarget\nblue\n\nCrate\norange\n\nWall\ngray\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nO = Target\n* = Crate\n# = Wall\nBoth = Player and Target\nStuff = Crate or Wall or Both\nAlias = Stuff\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Crate, Wall\n\n======\nRULES\n======\n\n[ Alias ] -> [ Alias ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 33 : Cannot define a property (something defined in terms of 'or') in terms of an aggregate (something that uses 'and'). In this case, you can't define \"STUFF\" in terms of \"BOTH\"."],1]
    ],
    [
        "mixed and/or legend used through a property should error instead of crashing (toUpperCase)",
        ["title mixed and/or used through a property in a rule\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = Crate\n# = Wall\nBlock = Crate or Wall and Player\nMoveable = Player or Block\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate, Wall\n\n======\nRULES\n======\n\n[ > Moveable | Moveable ] -> [ > Moveable | > Moveable ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 28 : Hey! You can't go mixing ANDs and ORs in a single legend entry.","line 29 : You're talking about BLOCK but it's not defined anywhere.","line 46 : Oops! BLOCK not assigned to a layer."],3]
    ],
    [
        "mixed and/or legend used through a synonym should error instead of crashing (layer)",
        ["title mixed and/or used through a synonym in a rule\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = Crate\n# = Wall\nStuff = Crate or Wall and Player\nAlias = Stuff\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate, Wall\n\n======\nRULES\n======\n\n[ Alias ] -> [ Alias ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 28 : Hey! You can't go mixing ANDs and ORs in a single legend entry.","line 29 : You're talking about STUFF but it's not defined anywhere.","line 46 : Oops! STUFF not assigned to a layer."],3]
    ],
    [
        "no on a synonym of an undefined mixed and/or name should error instead of crashing (objectMasks)",
        ["title mixed and/or used through a synonym in a rule\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = Crate\n# = Wall\nStuff = Crate or Wall and Player\nAlias = Stuff\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate, Wall\n\n======\nRULES\n======\n\n[ Alias no Alias no Alias ] -> [ Alias ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 28 : Hey! You can't go mixing ANDs and ORs in a single legend entry.","line 29 : You're talking about STUFF but it's not defined anywhere.","line 46 : You cannot specify the same object more than once in a single cell (in this case ALIAS occurs multiple times).","line 46 : Oops! STUFF not assigned to a layer."],4]
    ],
    [
        "MOVE sound on an aggregate should error instead of crashing (generateSoundData layer)",
        ["title sound on mixed-to-aggregate should error instead of crashing\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate1\nred\n\nCrate2\nblue\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nCrate = Crate1 and Crate2\n\n=======\nSOUNDS\n=======\n\nCrate MOVE 50239307\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate\n\n======\nRULES\n======\n\n[ > Player | Crate ] -> [ > Player | > Crate ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 39 : \"crate\" is an aggregate (defined using \"and\"), and cannot be added to a single layer because its constituent objects must be able to coexist.","line 26 : Object \"CRATE1\" has been defined, but not assigned to a layer.","line 26 : Object \"CRATE2\" has been defined, but not assigned to a layer.","line 45 : CRATE2 and CRATE1 can never overlap. This rule requires that to happen, so it impossible it will ever trigger.","line 14 : Object \"CRATE1\" has been defined, but not assigned to a layer.","line 17 : Object \"CRATE2\" has been defined, but not assigned to a layer.","line 32 : cannot assign sound events to aggregate objects (declared with \"and\"), only to regular objects, or properties, things defined in terms of \"or\" (\"crate\")."],7]
    ],
    [
        "aggregate defined in terms of properties should error instead of crashing (checkSuperfluousCoincidences)",
        ["title aggregate of properties should error instead of crashing\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate1\nred\n\nCrate2\nblue\n\nCrate3\ngreen\n\nCrate4\nyellow\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nCG1 = Crate1 or Crate2\nCG2 = Crate3 or Crate4\nCrate = CG1 and CG2\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate\n\n======\nRULES\n======\n\n[ > Player | Crate ] -> [ > Player | > Crate ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 34 : Cannot define an aggregate (using 'and') in terms of properties (something that uses 'or').","line 45 : \"crate\" is an aggregate (defined using \"and\"), and cannot be added to a single layer because its constituent objects must be able to coexist.","Trying to define aggregate \"CRATE\" in terms of property \"CG1\".","Trying to define aggregate \"CRATE\" in terms of property \"CG2\".","line 51 : Oops! CG1 not assigned to a layer.","line 51 : Oops! CG2 not assigned to a layer.","line 14 : Object \"CRATE1\" has been defined, but not assigned to a layer.","line 17 : Object \"CRATE2\" has been defined, but not assigned to a layer.","line 20 : Object \"CRATE3\" has been defined, but not assigned to a layer.","line 23 : Object \"CRATE4\" has been defined, but not assigned to a layer."],10]
    ],
    [
        "mixed and/or property used through random should error instead of crashing (rulesToMask)",
        ["title mixed and/or used with random should error instead of crashing\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = Crate\n# = Wall\nO = Crate or Wall\nN = Crate or Wall and Player\nN_O = N or O\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate, Wall\n\n======\nRULES\n======\n\n[ ] -> [ random N_O ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 29 : Hey! You can't go mixing ANDs and ORs in a single legend entry.","line 30 : You're talking about N but it's not defined anywhere."],2]
    ],
    [
        "background property with a missing member should error instead of crashing (generateExtraMembers layer)",
        ["title background property missing member should error instead of crashing\nauthor test\n\n========\nOBJECTS\n========\n\nBlack\nblack\n\nPlayer\nwhite\n\n=======\nLEGEND\n=======\n\n. = Black\nP = Player\nBackground = Black or White\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBlack\nPlayer\n\n======\nRULES\n======\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 20 : You're talking about WHITE but it's not defined anywhere."],1]
    ],
    [
        "movement on an undefined name should error instead of crashing (generateMatchString)",
        ["title movement on missing name should error instead of crashing\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\n[ > MissingName ] -> cancel\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",        ["line 36 : Name \"missingname\", referred to in a rule, does not exist.","line 36 : Error, malformed cell rule - was looking for cell contents, but found \"missingname\".  What am I supposed to do with this, eh, please tell me that.","line 36 : In a rule, if you specify a movement, it has to act on an object."],3]
    ],
    [
        "aggregate of missing objects should error instead of crashing (generateMasks ior)",
        ["title aggregate of missing objects should error instead of crashing\nauthor test\n\n========\nOBJECTS\n========\n\nPlayer\nwhite\n\n=======\nLEGEND\n=======\n\nP = Player\nBackground = Black and White\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nPlayer\n\n======\nRULES\n======\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 16 : You're talking about BLACK but it's not defined anywhere.","line 16 : You're talking about WHITE but it's not defined anywhere.","line 16 : background cannot be an aggregate (declared with 'and'), it has to be a simple type, or property (declared in terms of others using 'or')."],3]
    ],
    [
        "perpendicular as a rule direction should error instead of crashing (delta_index)",
        ["title perpendicular rule prefix\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = Crate\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate\n\n======\nRULES\n======\n\nperpendicular [ > Player | Crate ] -> [ > Player | > Crate ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP*\n",["line 40 : Name \"perpendicular\", referred to in a rule, does not exist.","line 40 : You cannot use relative directions (\"^v<>\") to indicate in which direction(s) a rule applies.  Use absolute directions indicators (Up, Down, Left, Right, Horizontal, or Vertical, for instance), or, if you want the rule to apply in all four directions, do not specify directions"],2]
    ],
    [
        "a win condition on an undefined name should error instead of crashing (bitsClearInArray)",
        ["title win on missing name\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\n==============\nWINCONDITIONS\n==============\n\nall Floop on Player\n\n=======\nLEVELS\n=======\n\nP\n",["line 40 : Error in win condition: \"FLOOP\" is not a valid object name.","line 40 : Unwelcome term \"floop\" found in win condition. I don't know what I'm supposed to do with this."],2]
    ],
    [
        "background property of missing objects should error instead of crashing (calcBackgroundMask)",
        ["title background property of missing objects\nauthor test\n\n========\nOBJECTS\n========\n\nLowPlayerTop\nwhite\n\nLowPlayer\nblack\n\n=======\nLEGEND\n=======\n\nP = LowPlayer\nBackground = LowFloor or Wall\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nLowPlayerTop\n\n======\nRULES\n======\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nP\n",["line 19 : You're talking about LOWFLOOR but it's not defined anywhere.","line 19 : You're talking about WALL but it's not defined anywhere.","Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!","Error, didn't find any object called player, either in the objects section, or the legends section.","you have to have a background layer","line 11 : Object \"LOWPLAYER\" has been defined, but not assigned to a layer."],6]
    ],
    [
        "trailing ellipsis with run_rules_on_level_start should error instead of crashing (matches)",
        ["title trailing ellipsis with run_rules_on_level_start\nrun_rules_on_level_start\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nPlayer\nwhite\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\n[ Player | ... ] -> [ Player | ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n.P\n",["line 36 : There's no point in putting an ellipsis at the very start or the end of a rule","line 36 : An ellipsis on the left must be matched by one in the corresponding place on the right."],2]
    ],
    [
        "aggregate Background with no assigned objects should error instead of crashing (generateExtraMembers id)",
        ["title aggregate Background with no assigned objects\nauthor test\n\n========\nOBJECTS\n========\n\n=======\nLEGEND\n=======\n\nBackground = Player and Wall\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nplayer\n\n======\nRULES\n======\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n",["line 12 : You're talking about PLAYER but it's not defined anywhere.","line 12 : You're talking about WALL but it's not defined anywhere.","line 22 : Cannot add \"PLAYER\" to a collision layer; it has not been declared.","Error, didn't find any object called player, either in the objects section, or the legends section. There must be a player!","You need to have some objects defined","line 12 : background cannot be an aggregate (declared with 'and'), it has to be a simple type, or property (declared in terms of others using 'or').","Error, didn't find any object called player, either in the objects section, or the legends section.","No levels found.  Add some levels!"],8]
    ],
    [
        "object Background redefined as an aggregate of missing names should error instead of crashing (getMaskFromName id)",
        ["OBJECTS\nBackground Player .\nLEGEND\nBackground = Player and Wall\nCOLLISIONLAYERS\nBackground\n",["line 4 : Name \"BACKGROUND\" already in use (on line line 2).","line 4 : You're talking about WALL but it's not defined anywhere.","line 5 : section \"COLLISIONLAYERS\" is out of order, must follow  \"SOUNDS\" (or it could be that the section \"SOUNDS\"is just missing totally.  You have to include all section headings, even if the section itself is empty).","line 2 : color not specified for object \"background\".","Error, didn't find any object called player, either in the objects section, or the legends section.","No levels found.  Add some levels!"],6]
    ],
    [
        "#1192 trailing comments on STARTLOOP are allowed",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\nstartloop (apply checkerboard for bridges)\n[ Player ] -> [ Player ]\nendloop\n\nWINCONDITIONS\nLEVELS\n\nP\n", [], 0]
    ],
    [
        "#1192 trailing nested comments on ENDLOOP are allowed",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\nSTARTLOOP\n[ Player ] -> [ Player ]\nENDLOOP (comment (nested comment))\n\nWINCONDITIONS\nLEVELS\n\nP\n", [], 0]
    ],
    [
        "#1192 comments inside rules still report an error",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n[ Player (comment) ] -> [ Player ]\n\nWINCONDITIONS\nLEVELS\n\nP\n", ["line 18 : You can't have comments inside rules, sorry. In PuzzleScript, '(' starts a comment, so everything after the first '(' on this line was ignored before your rule was parsed."], 1]
    ],
    [
        "#1192 STARTLOOP with a comment still requires ENDLOOP",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\nstartloop (missing endloop)\n[ Player ] -> [ Player ]\n\nWINCONDITIONS\nLEVELS\n\nP\n", ["line 18 : Yo I found a STARTLOOP without a corresponding ENDLOOP."], 1]
    ],
    [
        "diagnostic coverage: invalid object name",
        ["OBJECTS\n\nBackground .\nblack\n\n!\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n\n\nWINCONDITIONS\nLEVELS\n\nP\n",["line 6 : Unknown junk in object section (possibly: sprites have to be 5 pixels wide and 5 pixels high exactly. Or maybe: the main names for objects have to be words containing only the letters a-z0.9 - if you want to call them something like \",\", do it in the legend section)."],0]
    ],
    [
        "diagnostic coverage: ellipsis followed by an unbound movement",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n[ ... > ] -> [ Player ]\n\nWINCONDITIONS\nLEVELS\n\nP\n",["line 18 : Cannot end a rule with ellipses.","line 18 : Error, malformed cell rule - encountered a \"[\"\" before previous bracket was closed","line 18 : In a rule, each pattern to match on the left must have a corresponding pattern on the right of equal length (number of cells).","You have an totally empty pattern on the left-hand side.  This will match *everything*.  You certainly don't want this."],4]
    ],
    [
        "diagnostic coverage: movements without objects leave an empty pattern",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n[ > ] -> [ > ]\n\nWINCONDITIONS\nLEVELS\n\nP\n",["line 18 : In a rule, if you specify a movement, it has to act on an object.","line 18 : Error, malformed cell rule - encountered a \"[\"\" before previous bracket was closed","You have an totally empty pattern on the left-hand side.  This will match *everything*.  You certainly don't want this."],3]
    ],
    [
        "diagnostic coverage: random on a synonym of an undefined object",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nAlias = Missing\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n[ Player ] -> [ random Alias ]\n\nWINCONDITIONS\nLEVELS\n\nP\n",["line 10 : You're talking about MISSING but it's not defined anywhere.","line 19 : You want to spawn a random \"MISSING\", but I don't know how to do that"],2]
    ],
    [
        "diagnostic coverage: flickscreen dimension without an integer part",
        ["flickscreen .0x1\nOBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n\n\nWINCONDITIONS\nLEVELS\n\nP\n",["line 1 : Wasn't able to make sense of \".0\" as a dimension.","line 1 : Couldn't understand the dimensions given to me (you gave \".0x1\") - should be of the form AxB."],0]
    ],
    [
        "diagnostic coverage: zoomscreen dimension with a decimal exponent",
        ["zoomscreen .1e1x1\nOBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n\n\nWINCONDITIONS\nLEVELS\n\nP\n",["line 1 : Wasn't able to make sense of \".1e1\" as a dimension.","line 1 : Couldn't understand the dimensions given to me (you gave \".1e1x1\") - should be of the form AxB."],0]
    ],
    [
        "diagnostic coverage: sound event without a seed",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nsfx0\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n\n\nWINCONDITIONS\nLEVELS\n\nP\n",["line 11 : incorrect sound declaration."],0]
    ],
    [
        "diagnostic coverage: object sound without a verb or seed",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nPlayer\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n\n\nWINCONDITIONS\nLEVELS\n\nP\n",["line 11 : incorrect sound declaration."],0]
    ],
    [
        "#1207 randomdir is rejected on the left of a late rule",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nDust D\norange\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nDust\n\nRULES\n\nlate [ randomdir Dust ] -> [ Dust ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n.D.\n...\n", ["line 22 : Movements cannot appear in late rules."], 1]
    ],
    [
        "#1207 randomdir is rejected on the right of a late rule",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nDust D\norange\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nDust\n\nRULES\n\nlate [ Dust ] -> [ randomdir Dust ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n.D.\n...\n", ["line 22 : Movements cannot appear in late rules."], 1]
    ],
    [
        "#1207 randomdir remains allowed in non-late rules",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nDust D\norange\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nDust\n\nRULES\n\n[ Dust ] -> [ randomdir Dust ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n.D.\n...\n", [], 0]
    ],
    [
        "#1207 no remains allowed on both sides of late rules",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nDust D\norange\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nDust\n\nRULES\n\nlate [ Player no Dust ] -> [ Player no Dust ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n.D.\n...\n", [], 0]
    ],
    [
        "#1207 random rule groups remain allowed in late rules",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nDust D\norange\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nDust\n\nRULES\n\nlate random [ Dust ] -> [ ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n.D.\n...\n", [], 0]
    ],
    [
        "#1207 random object spawning remains allowed in late rules",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nDust D\norange\n\nLEGEND\nParticle = Player or Dust\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nDust\n\nRULES\n\nlate [ Dust ] -> [ random Particle ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n.D.\n...\n", [], 0]
    ],
    [
        "Alias names: legend names can start with punctuation and be used in rules",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nTarget T\nred\n\nLEGEND\n!a = Player or Target\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nTarget\n\nRULES\n[ !a ] -> [ !a ]\nWINCONDITIONS\nLEVELS\n\nP\n", [], 0]
    ],
    [
        "Alias names: legend names can end with punctuation and be used in rules",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\na! = Player\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n[ a! ] -> [ a! ]\nWINCONDITIONS\nLEVELS\n\nP\n", [], 0]
    ],
    [
        "Alias names: hyphenated legend properties can be used in rules",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nTarget T\nred\n\nLEGEND\na-b = Player or Target\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nTarget\n\nRULES\n[ a-b ] -> [ a-b ]\nWINCONDITIONS\nLEVELS\n\nP\n", [], 0]
    ],
    [
        "Alias names: object synonyms can contain punctuation and be used in rules",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P !a a! a-b\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n[ !a ] -> [ a! ]\n[ a-b ] -> [ a-b ]\nWINCONDITIONS\nLEVELS\n\nP\n", [], 0]
    ],
    [
        "Alias names: normal names retain Unicode letters, numbers, and underscores",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P Player_2 Caf\u00e9_3\nwhite\n\nTarget T\nred\n\nLEGEND\nPlayer_Target = Player_2 or Target\n^ = Caf\u00e9_3 or Player_Target\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nTarget\n\nRULES\nWINCONDITIONS\nLEVELS\n\nP\n", [], 0]
    ],
    [
        "Keyword aliases: single-character legend definitions are allowed",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nTarget\nred\n\nLEGEND\nV = Player\n^ = Player\n< = Player\n> = Player\n| = Player\n[ = Player and Target\n] = Player or Target\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nTarget\n\nRULES\nWINCONDITIONS\nLEVELS\n\nV^<>|[\n", [], 0]
    ],
    [
        "Keyword aliases: single-character object synonyms are allowed",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P V ^ < > | [ ]\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\nWINCONDITIONS\nLEVELS\n\nPV^<>|[]\n", [], 0]
    ],
    [
        "Keyword aliases: v can define a property used in win conditions",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nWall W\nred\n\nLEGEND\nv = Player or Wall\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer, Wall\n\nRULES\nWINCONDITIONS\nSome v\nLEVELS\n\nP\n", [], 0]
    ],
    [
        "Keyword aliases: longer object synonyms still warn",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P action cantmove sfx0\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\nWINCONDITIONS\nLEVELS\n\nP\n", ["line 6 : You named an object \"ACTION\", but this is a keyword. Don't do that!", "line 6 : You named an object \"CANTMOVE\", but this is a keyword. Don't do that!", "line 6 : You named an object \"SFX0\", but this is a keyword. Don't do that!"], 0]
    ],
    [
        "Keyword aliases: movement aliases still warn and cannot replace objects in rules",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P action\nwhite\n\nWall W\nred\n\nLEGEND\n\nSOUNDS\n\nCOLLISIONLAYERS\nBackground\nPlayer, Wall\n\nRULES\n[ action ] -> [ ]\nWINCONDITIONS\nLEVELS\n\nP.W\n", ["line 6 : You named an object \"ACTION\", but this is a keyword. Don't do that!", "line 21 : In a rule, if you specify a movement, it has to act on an object.", "line 21 : Error, malformed cell rule - encountered a \"[\"\" before previous bracket was closed", "line 21 : In a rule, each pattern to match on the left must have a corresponding pattern on the right of equal length (number of cells).", "You have an totally empty pattern on the left-hand side. This will match *everything*. You certainly don't want this."], 4]
    ],
    [
        "Keyword aliases: sound event aliases still warn and cannot replace objects in sounds",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P sfx0\nwhite\n\nWall W\nred\n\nLEGEND\n\nSOUNDS\nsfx0 move 12345\nCOLLISIONLAYERS\nBackground\nPlayer, Wall\n\nRULES\n\nWINCONDITIONS\nLEVELS\n\nP.W\n", ["line 6 : You named an object \"SFX0\", but this is a keyword. Don't do that!", "line 15 : Was expecting a sound seed here (a number like 123123, like you generate by pressing the buttons above the console panel), but found something else."], 1]
    ],
    [
        "Keyword aliases: primary object names still warn",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nV\nred\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer, V\n\nRULES\nWINCONDITIONS\nLEVELS\n\nPV\n", ["line 9 : You named an object \"V\", but this is a keyword. Don't do that!"], 0]
    ],
    [
        "Keyword aliases: longer legend names still warn",
        ["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\naction = Player\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\nWINCONDITIONS\nLEVELS\n\nP\n", ["line 10 : You named an object \"ACTION\", but this is a keyword. Don't do that!"], 0]
    ],
];
