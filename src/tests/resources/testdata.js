//new tests get appended to the end innit
var testdata = [
	[
		"sokoban no win condition",
		["(\nSimple block pushing example, simplified\n)\n\n========\nOBJECTS\n========\n\nBackground\nBlack\n\nTarget \nYELLOW\n\nPlayer \nPINK \n\nCrate\nRED\n\nWall \nBROWN DARKBROWN \n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\n@ = Crate and Target\nO = Target\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES \n====== \n\n[ > Player | Crate ] -> [ > Player | > Crate ] \n\n==============\nWINCONDITIONS\n==============\n\n======= \nLEVELS\n=======\n\n#########\n#.......#\n#.....@.#\n#.P.*.O.#\n#.......#\n#.......#\n#########\n\n#########\n#.....@.#\n#.P.*.O.#\n#.......#\n#########\n\n#########\n#.....@.#\n#.P.*.O.# \n#.......#\n#########", [0, 0, 3, 3, 3, 3, 3, 2, 1, 0, 1, 2, 1, 1, 2, 3, 3, 2, 3, 0, 1, 0, 3, 0, 3, 2, 1, 2, 3, 1, 2, 3], "background wall:0,0,0,0,0,0,0,0,background:1,\n1,1,1,1,0,0,1,1,1,\n1,1,0,0,1,1,1,1,1,\n0,0,1,1,1,1,1,0,0,\n1,1,background player:2,1,1,0,0,1,background crate target:3,\n3,1,1,0,0,1,1,1,1,\n1,0,0,0,0,0,0,0,0,\n"]
	],
	[
		"sokoban with win condition",
		["(\nSimple block pushing example, simplified\n)\n\n========\nOBJECTS\n========\n\nBackground\nBlack\n\nTarget \nYELLOW\n\nPlayer \nPINK \n\nCrate\nRED\n\nWall \nBROWN DARKBROWN \n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\n@ = Crate and Target\nO = Target\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES \n====== \n\n[ > Player | Crate ] -> [ > Player | > Crate ] \n\n==============\nWINCONDITIONS\n==============\n\nall crate on target\n\n======= \nLEVELS\n=======\n\n#########\n#.......#\n#.....@.#\n#.P.*.O.#\n#.......#\n#.......#\n#########\n\n#########\n#.....@.#\n#.P.*.O.#\n#.......#\n#########\n\n#########\n#.....@.#\n#.P.*.O.# \n#.......#\n#########", [0, 0, 3, 3, 3, 3, 3, 2, 1, 0, 1, 1, 2, 3, 1, 1, 2, 3, 3], "background wall:0,0,0,0,0,0,background:1,1,1,\n0,0,1,background player:2,1,0,0,1,1,\n1,0,0,1,background crate:3,1,0,0,1,\n1,1,0,0,background crate target:4,background target:5,1,0,0,\n1,1,1,0,0,0,0,0,0,\n"]
	],
	[
		"block faker no win condition",
		[{file:"test_scripts/block_faker_no_win_condition.txt"}, [3, 0, 1, 0, 3, 3, 3, 0, 3, 2, 2, 3, 3, 3], "background wall:0,0,0,0,0,0,0,background orangeblock:1,0,0,\nbackground grille:2,2,2,0,0,0,0,0,background:3,3,\n2,0,0,0,0,0,3,0,0,0,\n0,0,0,3,3,3,0,0,0,0,\n0,3,3,3,3,3,3,0,0,3,\n0,3,0,0,0,0,0,0,2,2,\n2,0,0,0,0,0,0,background endpoint player:4,2,0,\n0,0,0,0,0,0,0,0,0,0,\n"]
	],
	[
		"block faker with win condition",
		[{file:"test_scripts/block_faker_with_win_condition.txt"}, [3, 0, 1, 0, 3, 3, 3, 0, 3, 2, 2, 3, 3, 3, 3, 2, 3, 2, 2, 2, 1, 1, 3, 3, 0, 0, 0, 1, 1, 2, 2, 0, 3, 3, 2, 2, 3, 2, 3, 3, 2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3, 3, 0, 0, 1, 2, 2, 3, 2, 1, 1, 2, 1, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], {file:"test_endstates/block_faker_with_win_condition.txt"}]
	],
	[
		"by your side",
		[{file:"test_scripts/by_your_side.txt"}, [1, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 2, 1, 1, 0, 1, 3, 3, 0, 1, 1, 1, 2, 2, 1, 1, 3, 2, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 1, 0, 0, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 0, 0, 3, 1, 2, 3, 3, 3, 3, 3, 3, 1, 0, 1, 1, 2, 3, 3, 3, 3, 0, 1, 1, 1, 1, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 3, 3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 0, 0, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 3, 0, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 0, 0, 0, 1, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 0, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 0, 1, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3, 3, 2, 2, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 0, 0, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 2, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 0, 3, 0, 1, 0, 1, 1, 1, 3, 3, 3, 2, 3, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 3, 2, 3, 3, 0, 3, 0, 3, 3, 3, 0, 3, 3, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 3, 0, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 0, 1, 1, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 0, 0, 2, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 2, 3, 3, 0, 3, 0, 3, 3, 3, 0, 2, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3, 2, 3, 3, 0, 3, 0, 3, 3, 3, 3, 1, 0, 0, 1, 1, 1, 0, 1, 0, 3, 2, 3, 3, 3, 2, 3, 2, 2, 1, 1, 0, 1, 1, 1, 1, 2, 1, 2, 1, 0, 1, 0, 1, 0, 3, 3, 0, 3, 3, 3, 0, "undo", "undo", "undo", 3, 0, 2, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 1, 0, 1, 0, 1, 1, 0, 1, 2, 2, 1, 1, 0, 0, 0, 0, 3, 3, 3, 2, 2, 2, 1, 2, 1, 1, 1, 0, 0, 0, 0, 3, 3, 2, 3, 2, 2, 2, 1, 1, 1, 0, 1, 2], {file:"test_endstates/by_your_side.txt"}]
	],
	[
		"ellipsisTest1",
		["(\nSimple block pushing example, simplified\n)\n\n========\nOBJECTS\n========\n\nBackground\ngray\n\nTarget \nYELLOW\n\nPlayer \nPINK \n\nCrate\nRED\n\nWall \nBROWN DARKBROWN \n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\n@ = Crate and Target\nO = Target\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES \n====== \n\n[ > Player | ... | Crate ] -> [ > Player | ... | > Crate ] \n\n==============\nWINCONDITIONS\n==============\n\n======= \nLEVELS\n=======\n\n...............\n...............\n...............\n...............\n...............\n...............\n..P...*........\n...............\n...............\n...............\n...............\n...............\n...............\n\n", [3, 3, 3, 2, 2, 3, 3, 3, 3, 0, 0, 3, 3, 3, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1], {file:"test_endstates/ellipsistest1.txt"}]
	],
	[
		"ellipsisPropagationBug1",
		["debug\n\n========\nOBJECTS\n========\n\nBackground\ndarkgray\n\nMoveGuide\npink\n\nPlayer\nred\n\n=======\nLEGEND\n=======\n. = Background\nP = player\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nMoveGuide\nPlayer\n\n======\nRULES \n====== \n\nUP [ Player | ... | ] -> [ Player | ... | DOWN MoveGuide ]\nLEFT [ Player | ... | ] -> [ Player | ... |RIGHT MoveGuide ]\n\n==============\nWINCONDITIONS\n==============\n\n\n======= \nLEVELS\n=======\n\n...\n...\n..p", [1, 0, 1, 0], "background moveguide player:0,background moveguide:1,1,\n1,1,1,\n1,1,1,\n"]
	],
	[
		"ellipsisPropagationBug2",
		["(\nSimple block pushing example, simplified\n)\n\n========\nOBJECTS\n========\n\n(temporary objects, for programming)\n\nMoveGuide\npink\n\nBackground\ndarkgray\n\n\nPlayer P\nRED BLACK white\n.2.2.\n.000.\n01110\n00100\n00100\n\n\n=======\nLEGEND\n=======\n. = Background\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nMoveGuide\nPlayer\n\n======\nRULES \n====== \n\n(movement manager code begin)\n\n(seed move guides)\nUP [Player | ... |] -> [ Player | ... | DOWN MoveGuide ]\nLEFT [Player | ... |] -> [ Player | ... |RIGHT MoveGuide ]\n\n\nVERTICAL [ Horizontal MoveGuide | ] -> [ Horizontal MoveGuide | Horizontal MoveGuide ]\n\n\n\n==============\nWINCONDITIONS\n==============\n\n\n======= \nLEVELS\n=======\n\n....\n....\n....\n...p", [0, 1, 0, 1, 0, 1], "background moveguide player:0,background moveguide:1,1,1,\n1,1,1,1,\n1,1,1,1,\n1,1,1,1,\n"]
	],
	[
		"undo test",
		["(\nSimple block pushing example, simplified\n)\n\n========\nOBJECTS\n========\n\nBackground\nBlack\n\nTarget \nYELLOW\n\nPlayer \nPINK \n\nCrate\nRED\n\nWall \nBROWN DARKBROWN \n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\n@ = Crate and Target\nO = Target\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES \n====== \n\n[ > Player | Crate ] -> [ > Player | > Crate ] \n\n==============\nWINCONDITIONS\n==============\nall target on crate\n\n======= \nLEVELS\n=======\n\n#########\n#.......#\n#.....@.#\n#.P.*.O.#\n#.......#\n#.......#\n#########\n\n#########\n#.....@.#\n#.P.*.O.#\n#.......#\n#########\n\n#########\n#.....@.#\n#.P.*.O.# \n#.......#\n#########", [3, 0, 3, "undo", 1, "undo", "undo", 2, 3, 2], "background wall:0,0,0,0,0,0,0,0,background:1,\n1,1,1,1,0,0,1,1,1,\n1,1,0,0,1,1,1,1,1,\n0,0,1,1,background crate:2,1,background player:3,0,0,\n1,1,1,1,1,0,0,1,background crate target:4,\nbackground target:5,1,1,0,0,1,1,1,1,\n1,0,0,0,0,0,0,0,0,\n"]
	],
	[
		"restart test",
		["(\nSimple block pushing example, simplified\n)\n\n========\nOBJECTS\n========\n\nBackground\nBlack\n\nTarget \nYELLOW\n\nPlayer \nPINK \n\nCrate\nRED\n\nWall \nBROWN DARKBROWN \n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\n@ = Crate and Target\nO = Target\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES \n====== \n\n[ > Player | Crate ] -> [ > Player | > Crate ] \n\n==============\nWINCONDITIONS\n==============\nall target on crate\n\n======= \nLEVELS\n=======\n\n#########\n#.......#\n#.....@.#\n#.P.*.O.#\n#.......#\n#.......#\n#########\n\n#########\n#.....@.#\n#.P.*.O.#\n#.......#\n#########\n\n#########\n#.....@.#\n#.P.*.O.# \n#.......#\n#########", [0, 0, 3, 3, 2, 2, 2, "restart", 2, 2, 3, 3, 0, 0, 1, 0, 3, 0, 3, 2, 1, 2, 3, 0, 3, 3, "restart", 2, 2, 3, 3, 3, "restart", 3, 0, 3, 3, 2, 2, 3, 3, 0, 0, 1, 2, 2, 1, 1, "restart", 3, 3, 2, 3, 3, 3, 0, 2, 2, 2, 2, 1, 1, 2, 1, 0, 3], "background wall:0,0,0,0,0,0,background:1,1,1,\n0,0,1,background player:2,1,0,0,1,1,\n1,0,0,1,background crate:3,1,0,0,1,\n1,1,0,0,background crate target:4,background target:5,1,0,0,\n1,1,1,0,0,0,0,0,0,\n"]
	],
	[
		"cat bastard",
		[{file:"test_scripts/cat_bastard.txt"}, [2, 3, 3, 3, 0, 3, 3, 1, 1, 1, 1, 3, 3, 3, 3, 0, 3, 3, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1, 1, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 0], "background wall:0,0,0,0,0,0,0,0,0,0,0,0,\nbackground cratespoken:1,0,0,background:2,1,2,0,0,1,0,0,2,\n0,0,2,2,2,0,0,2,0,0,2,2,\n1,0,2,0,1,background player:3,2,0,0,2,0,2,\n2,2,2,2,0,2,0,0,2,0,0,0,\n2,0,0,0,2,0,0,2,2,0,2,2,\n2,0,2,2,0,0,2,2,2,2,1,2,\n2,2,2,0,0,2,2,2,2,2,2,2,\n2,2,0,0,2,2,2,2,2,2,2,2,\n2,0,0,2,2,2,2,2,2,2,2,2,\n0,0,0,0,0,0,0,0,0,0,0,0,\n"]
	],
	[
		"rule grouping test",
		[{file:"test_scripts/rule_grouping_test.txt"}, [3, 3, 3], "background:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,background player:1,0,0,background bluecrate:2,0,0,background redcrate:3,0,0,2,0,0,3,0,\n0,2,0,0,3,0,0,2,0,0,3,0,0,0,0,\n"]
	],
	[
		"simple rigid test",
		["========\nOBJECTS\n========\n\nBackground\ngray\n\nTarget \nYELLOW\n\nPlayer \nPINK \n\nCrate w\nGreen\n\n\nWall \nBROWN DARKBROWN \n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\nO = Target\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES \n====== \n\n[ > Player | Crate ] -> [ > Player | > Crate ] \n+ rigid [ left Crate | Crate ] -> [ left Crate | left Crate ]\n\n==============\nWINCONDITIONS\n==============\n\n======= \nLEVELS\n=======\n\n.wp\n.w.\n\n", [1], "background crate:0,0,background player:1,\nbackground:2,2,2,\n", 0]
	],
	[
		"rigid body test",
		["(\nSimple block pushing example, simplified\n)\n\n========\nOBJECTS\n========\n\nBackground\ngray\n\nTarget \nYELLOW\n\nPlayer \nPINK \n\nCrate\nRED\n\nWall \nBROWN DARKBROWN \n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\n@ = Crate and Target\nO = Target\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES \n====== \n\n[ > Player | Crate ] -> [ > Player | > Crate ] \n+ rigid [ moving Crate | Crate ] -> [ moving Crate | moving Crate ]\n\n==============\nWINCONDITIONS\n==============\n\n======= \nLEVELS\n=======\n\n...............\n....*..........\n....*..........\n...............\n.....***.......\n.....*.........\n..P..**#.......\n.....*.........\n...............\n...............\n...............\n...............\n...............\n\n", [2, 0, 0, 0, 3, 3, 0, 3, 2, 2, 2, 1, 1, 0, 0, 3, 2, 1], {file:"test_endstates/rigid_body_test.txt"}]
	],
	[
		"rigid body test 2",
		[{file:"test_scripts/rigid_body_test_2.txt"}, [0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 2, 2, 3, 3, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 3, 3, 3, 1, 1, 1, 1, 2, 3, 0, 3, 3, 3, 3, 2, 1, 0, 1, 2, 0, 1, 1, 2, 2, 2, 1, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 0, 3, 0, 3, 0, 0, 1, 0, 1, 1, 1, 1, 2, 1, 2, 3, 0, 3, 3, 3, 3, 3, 1, 2, 2, 3, 2, 1, 1, 1, 1, 1, 2, 1, 0, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 0, 2, 1, 2, 3, 2, 1, 1, 1, 1, 1, 1, 2, 1, 0, 3, 3, 3, 3, 3, 3, 0, 0, 0, 1, 3, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 1, 0, 0, 3], {file:"test_endstates/rigid_body_test_2.txt"}]
	],
	[
		"annoying edge case",
		["(\nSimple block pushing example, simplified\n)\n\n========\nOBJECTS\n========\n\n(temporary objects, for programming)\n\nMoveGuide\npink\n\n(game objects)\n\nBackground\ndarkgray\n\n\nPlayer P\nRED BLACK white\n.2.2.\n.000.\n01110\n00100\n00100\n\n\n=======\nLEGEND\n=======\nU = MoveGuide\n. = Background\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\nMoveGuide\n\n======\nRULES \n====== \n\n(movement manager code begin)\n\n(seed move guides)\nRIGHT [Player | ... | ] -> [ Player | ... |LEFT MoveGuide ]\nLEFT [Player | ... | ] -> [ Player | ... |RIGHT MoveGuide ]\n\n==============\nWINCONDITIONS\n==============\n\n\n======= \nLEVELS\n=======\n\n.P.\n", [0], "background:0,background moveguide player:1,background moveguide:2,\n"]
	],
	[
		"propagation test",
		[{file:"test_scripts/propagation_test.txt"}, [1, 0, 1, 0, 1], "background moveguide:0,0,0,background:1,1,0,0,0,1,1,0,0,0,\n1,1,background moveguide player:2,0,0,1,1,0,background moveguide2:3,3,3,3,3,\n3,3,3,3,3,3,3,3,3,3,3,3,1,\n1,3,3,3,1,1,3,3,3,1,1,3,3,\n3,1,1,3,3,3,1,1,3,3,3,1,1,\n"]
	],
	[
		"test testing starting at level N",
		[{file:"test_scripts/test_testing_starting_at_level_n.txt"}, [2, 2, 2, 3, 3, 0, 0], {file:"test_endstates/test_testing_starting_at_level_n.txt"}, 8]
	],
	[
		"beginloop/endloop test",
		[{file:"test_scripts/beginloop_endloop_test.txt"}, [3], "background:0,0,background player:1,background bluecrate:2,background redcrate:3,3,\n", 0]
	],
	[
		"beginloop/endloop test #2 (w/ rigids)",
		[{file:"test_scripts/beginloop_endloop_test_2_w_rigids.txt"}, [1], "background:0,background redcrate:1,1,1,0,0,1,background greencrate:2,0,\n0,0,0,2,0,0,0,2,2,\n0,0,0,background player:3,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n", 0]
	],
	[
		"late beginloop/endloop test",
		[{file:"test_scripts/late_beginloop_endloop_test.txt"}, [3, 3, 3, 2, 2], {file:"test_endstates/late_beginloop_endloop_test.txt"}, 1]
	],
	[
		"multiple loops",
		[{file:"test_scripts/multiple_loops.txt"}, [3, 3, 1, 0, 3, 1, 2, 2, 3], "background:0,0,0,0,0,background player:1,0,0,background crate2:2,background crate1:3,background crate3:4,4,2,background crate4:5,\n2,3,4,4,2,5,2,3,4,4,2,5,2,3,\n4,4,2,5,0,0,0,0,0,0,0,0,0,0,\n", 0]
	],
	[
		"beginloop/endloop with mutual recursion",
		["title Mutual Recursion Test\nauthor Scott Hughes\n\n=======\nOBJECTS\n=======\n\nBackground\nBLACK\n\nPlayer\nBLUE\n\nCrate\nORANGE\n\n======\nLEGEND\n======\n\n. = Background\nP = Player\nC = Crate\n\n======\nSOUNDS\n======\n\n===============\nCOLLISIONLAYERS\n===============\n\nBackground\nPlayer, Crate\n\n=====\nRULES\n=====\n\nstartloop\n\n  [ Player ] -> [ Crate ]\n  [ Crate ] -> [ Player ]\n\nendloop\n\n=============\nWINCONDITIONS\n=============\n\n======\nLEVELS\n======\n\n...\n.P.\n...\n", [0], "background:0,0,0,\nbackground player:1,0,0,\n0,0,0,\n", 0]
	],
	[
		"ellipsis magnetism test",
		[{file:"test_scripts/ellipsis_magnetism_test.txt"}, [3, 3, 2, 2, 2, 4], "background floor goal:0,background floor:1,1,1,1,1,1,\n1,1,1,1,1,1,1,\n1,1,1,1,1,1,1,\n1,1,1,1,1,1,1,\n1,1,1,1,1,1,background crate:2,\n1,1,1,1,1,1,background playerattract:3,\n1,1,1,1,1,1,1,\n", 0]
	],
	[
		"optimization that broke andrew's game",
		[{file:"test_scripts/optimization_that_broke_andrew_s_game.txt"}, [1], "lowfloorraw:0,0,lowfloorraw lowplayer:1,0,\n", 0]
	],
	[
		"simple sokobond test",
		["title Sokobond\nauthor Alan Hazelden, Port by Jonah Ostroff\ntext_color #444444\nbackground_color #EEEEEE\nkey_repeat_interval 0.25\nrun_rules_on_level_start\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground \n#FFFFFF #EEEEEE\n00000\n01010\n00000\n01010\n00000\n\nPlayer\nPink\n00000\n0...0\n0...0\n0...0\n00000\n\nAtom\nRed Black\n.....\n.111.\n.101.\n.111.\n.....\n\n=======\nLEGEND\n=======\n\n. = Background\n1 = Atom and Player \nC = Atom\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nAtom\nPlayer\n\n\n======\nRULES \n======\n\nright [ right Player Atom] -> [ right Player right Atom]\n\nRight [ right Atom | Atom] -> [ Right Atom | Right Atom]\n\n\n==============\nWINCONDITIONS\n==============\n\n======= \nLEVELS\n=======\n\n1C.\n\n", [3], "background:0,atom background player:1,atom background:2,\n", 0]
	],
	[
		"loop length 1",
		[{file:"test_scripts/loop_length_1.txt"}, [3], "background player:0,0,0,0,0,0,\n0,0,0,0,background player target:1,0,\n0,0,0,1,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0,\n", 0]
	],
	[
		"loop length 1-b",
		[{file:"test_scripts/loop_length_1_b.txt"}, [3], "background target:0,0,0,0,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0,\n0,0,0,0,0,0,\n", 0]
	],
	[
		"mirror loop test",
		[{file:"test_scripts/mirror_loop_test.txt"}, [4], {file:"test_endstates/mirror_loop_test.txt"}, 0]
	],
	[
		"mirror + test",
		[{file:"test_scripts/mirror_test.txt"}, [4], {file:"test_endstates/mirror_loop_test.txt"}, 0]
	],
	[
		"collapse simple",
		[{file:"test_scripts/collapse_simple.txt"}, [3, 3, 3, 0, 3, 3, 1, 0, 3, 0, 1, 0, 3, 0, 3, 1, 1, 0, 1, 0, 1, 1, "undo", 0, 1], {file:"test_endstates/collapse_simple.txt"}, 3]
	],
	[
		"collapse long",
		[{file:"test_scripts/collapse_simple.txt"}, [3, 3, 0, 3, 3, 1, 0, 3, 0, 3, 1, 0, 1, 0, 1, 0, 1, 0, 3, 3, 3, 3, 1, 0, 3, 0, 3, "undo", 1, 3, 3, 0, 3, 0, 1, 3, 3, 3, 1, 0, 1, 0, 3, 1, 3, 3, 3, 0, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 1, 0, 1, 0, 3, 3, 1, 1, 1, 0, 1, 1, 3, 0, 1, 0, 1, "undo", 2, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 3, 3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 3, 0, 0, 1, 0, 1, "undo", 2], {file:"test_endstates/collapse_long.txt"}, 3]
	],
	[
		"unnecessary number of rules sokobond",
		[{file:"test_scripts/unnecessary_number_of_rules_sokobond.txt"}, [0, 2, 2, "undo", 1, 2, 3, 2, 1, 0, 0, 3, 0, 3, 0, 0, 1, 1, 1, 1], "background wall:0,0,0,0,0,0,0,0,0,\n0electron background helium:1,background:2,0,0,0,0,0,0,0electron background bondr hydrogen:3,\n1,0,0,0,0,0,0,0electron background bondl hydrogen player:4,2,\n2,2,2,2,0,0,2,2,2,\n2,0electron background bondd hydrogen:5,2electron background bondu nitrogen:6,0,0,2,2,2,2,\n2,2,0,0,2,2,0,0,0,\n0,0,0,2,1,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n", 34]
	],
	[
		"hos2",
		[{file:"test_scripts/hos2.txt"}, [4, 2, 4, 2, 1, 1, 4, 2, 2, 4, 2, 2, 3, 3, 0, 3, 3, 2, 3, 3, 4, 4, 0, 4, 2, 2, 4, 1, 1, 2, 2, 2, 1, 1, 0, 0, 1, 4, 2], "background wall:0,0,0,0,0,0,0,0,0,0,background:1,1,\n1,background goal:2,1,1,1,0,0,0,0,0,background sthief yellowopen:3,0,\n0,0,0,0,1,1,1,1,1,1,1,0,\n0,1,background crate yellowswitch:4,1,1,1,4,1,0,0,1,1,\n1,1,1,1,1,0,0,1,1,1,4,1,\n1,1,0,0,1,1,1,1,1,1,1,0,\n0,0,1,0,1,0,1,0,0,0,1,1,\n1,1,1,1,background goal priest:5,0,0,0,background dragonleft:6,1,6,1,\n6,0,0,0,0,0,0,0,0,0,0,0,\n", 27]
	],
	[
		"damn I'm huge",
		[{file:"test_scripts/damn_i_m_huge.txt"}, [1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 3, 3, 2, 1, 1, 1, 2, 3, "restart", 1, 1, 1, 0, 0, 3, 3, 0, 0, 0, 1, 3, 2, 2, 2, 1, 1, 1, 0, 3, 0, 0, 3, 1, "undo", "undo", "undo", "undo", 0, 0, "undo", 3, 3, 0, 1, 0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 3, 1, 1, 3, 2, 2, 2, 1, 1, 0, 0, 3, 3, 3, 3, 0, 0, 2, 1, 1, 2, 1, 0, 3, 3, 3, 0, 1, 1, 0, 2, 2, 2, 1, 2, 1, 1, 1, 1, 3, 2, 2, 2], {file:"test_endstates/damn_i_m_huge.txt"}, 13]
	],
	[
		"rule application hat test",
		["title Tiny but with massive hats\nauthor Weeble\nhomepage www.puzzlescript.net\nrun_rules_on_level_start\n\n========\nOBJECTS\n========\n\nLowFloor\nDARKGREY\n\nWall\n#bbbbbb\n\nLowPlayerTop\nWHITE BLACK\n.....\n.111.\n.111.\n.111.\n.111.\n\nLowPlayer\nWHITE BLACK\n11111\n.000.\n.000.\n.0.0.\n.....\n\n\n=======\nLEGEND\n=======\n\n. = LowFloor\n# = Wall\nP = LowPlayer\nPlayer = LowPlayer\nBackground = LowFloor or Wall\n\nTop = LowPlayerTop\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nLowPlayer, Wall\nLowPlayerTop\n\n======\nRULES\n======\n\n(Destroy all tops. We'll recreate them at the end after movement.)\n[ Top ] -> []\n\n(Re-assemble tops)\nlate down [ no LowPlayerTop | LowPlayer ] -> [ LowPlayerTop | LowPlayer ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n.\nP\n\n", [1], "lowfloor lowplayertop:0,\nlowfloor lowplayer:1,\n", 0]
	],
	[
		"ortho test 1",
		["title orthogonal rule test\n\n========\nOBJECTS\n========\n\nBackground .\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\nPlayer p\nPink\n\nDiamond \nBlue\n.....\n.....\n..0..\n.....\n.....\n\nCrate\nOrange\n00000\n0...0\n0...0\n0...0\n00000\n\n=======\nLEGEND\n=======\n=======\nSOUNDS\n=======\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\nCrate\nDiamond\n\n======\nRULES\n======\n\n[ Diamond ] -> [ ]\n[ Crate ] -> [ ]\n\n[ Action Player ] [ ] -> [ Action Player ] [ Diamond ]\n[ ORTHOGONAL Player ] [ ] -> [ ORTHOGONAL Player ] [ Crate ]\n\n==============\nWINCONDITIONS\n==============\n=======\nLEVELS\n=======\n\nmessage press x to incorrectly trigger orthogonal rule?\n.....\n.....\n..p..\n.....\n.....\n\n", [4], "background diamond:0,0,0,0,0,\n0,0,0,0,0,\n0,0,background diamond player:1,0,0,\n0,0,0,0,0,\n0,0,0,0,0,\n", 1]
	],
	[
		"ortho test 2",
		["title orthogonal rule test\n\n========\nOBJECTS\n========\n\nBackground .\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\nPlayer p\nPink\n\nDiamond \nBlue\n.....\n.....\n..0..\n.....\n.....\n\nCrate\nOrange\n00000\n0...0\n0...0\n0...0\n00000\n\n=======\nLEGEND\n=======\n=======\nSOUNDS\n=======\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\nCrate\nDiamond\n\n======\nRULES\n======\n\n[ Diamond ] -> [ ]\n[ Crate ] -> [ ]\n\n[ Action Player ] [ ] -> [ Action Player ] [ Diamond ]\n[ ORTHOGONAL Player ] [ ] -> [ ORTHOGONAL Player ] [ Crate ]\n\n==============\nWINCONDITIONS\n==============\n=======\nLEVELS\n=======\n\nmessage press x to incorrectly trigger orthogonal rule?\n.....\n.....\n..p..\n.....\n.....\n\n", [1], "background crate:0,0,0,0,0,\n0,0,background crate player:1,0,0,\n0,0,0,0,0,\n0,0,0,0,0,\n0,0,0,0,0,\n", 1]
	],
	[
		"too many rigid bodies",
		[{file:"test_scripts/too_many_rigid_bodies.txt"}, [2, 2, 2, 3, 2, 1, 0, 1, 2, 1, 2, 3, 3, 0, 2, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 0, 3, 3, 2, 1, 1, 1, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 0, 3, 2, 3, 0, 0, 0, 0, 0], {file:"test_endstates/too_many_rigid_bodies.txt"}, 0]
	],
	[
		"don't mask movements if no movements happening",
		["title Shield Game\nauthor Alan Hazelden\nhomepage www.draknek.org\nrequire_player_movement\n\ndebug\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground \nGREEN \n\nPlayer \nBlue\n.....\n..0..\n.000.\n.0.0.\n.....\n\nShieldR\nDarkBlue\n.....\n....0\n....0\n....0\n.....\n\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player and ShieldR\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\nShieldR\n\n======\nRULES \n====== \n\nright [ right Player ShieldR ] -> [ right Player right ShieldR ]\n\nright [ > Player ShieldR ] -> [ > Player ShieldR ]\n\n==============\nWINCONDITIONS\n==============\n\n======= \nLEVELS\n=======\n\nP.\n\n", [3], "background:0,background player shieldr:1,\n", 0]
	],
	[
		"zen puzzle garden",
		[{file:"test_scripts/zen_puzzle_garden.txt"}, [3, 3, 2, 3, 3, 3, 3, 0, 0, 3, 0, 1, 1, 1], {file:"test_endstates/zen_puzzle_garden.txt"}, 1]
	],
	[
		"Remove movements from empty layers after rule application",
		["title Object Replacement Test Case\nauthor Scott Hughes\n\n=======\nOBJECTS\n=======\n\nBackground\nwhite\n\nPlayer\nblue\n\nCrate\nred\n\nSpawn\ngreen\n\n======\nLEGEND\n======\n\n. = Background\nP = Player and Spawn\n\n======\nSOUNDS\n======\n\n===============\nCOLLISIONLAYERS\n===============\n\nBackground\nSpawn\nPlayer, Crate\n\n=====\nRULES\n=====\n\n[ Player ] -> [ ]\n\n[ Spawn ] -> [ Spawn Crate ]\n\n=============\nWINCONDITIONS\n=============\n\n======\nLEVELS\n======\n\n...\n.P.\n...\n", [3], "background:0,0,0,\n0,background crate spawn:1,0,\n0,0,0,\n", 0]
	],
	[
		"movement matching - correctly matching different objects same cell moving in different directions",

		["verbose_logging\ndebug\n\n========\nOBJECTS\n========\n\nBackground\ndarkgreen\n\nPlayer\ngreen\n\nRed\nred\n\nBrown\nbrown\n\nBlue\nblue\n\nA\nwhite\n\nB\nyellow\n\n=======\nLEGEND\n=======\n\nr = Red\no = Brown\n. = Background\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\nA\nB\nRed, Brown\nBlue\n\n======\nRULES\n======\n\n[ Red ] -> [ down A down B ]\n[ Brown ] -> [ down A up B ]\n[ down A up B ] -> [ ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nro\n\n\n", [3], "a b background:0,background:1,\n", 0]
	],
	[
		"movement matching - ellipsis bug - forgot to include one case in above",
		["verbose_logging\ndebug\n\n========\nOBJECTS\n========\n\nBackground\ndarkgreen\n\nPlayer\ngreen\n\nRed\nred\n\nBrown\nbrown\n\nBlue\nblue\n\nA\nwhite\n\nB\nyellow\n\n=======\nLEGEND\n=======\n\nr = Red\no = Brown\n. = Background\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\nA\nB\nRed, Brown\nBlue\n\n======\nRULES\n======\n\n[ Red | ... | Brown ] -> [ down A down B | ... | down a up B]\n[ down A up B | ... | down A up b ] -> [ | ... | ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nro\n\n", [2], "a b background:0,0,\n", 0]
	],
	[
		"ellipsis bug: rule matches two candidates, first replacement invalidates second",
		["title ellipsis cloning\n(ellipsis cloning bug -- reduction from neko puzzle)\n\n========\nOBJECTS\n========\n\nBackground .\nDarkBlue\n\nFruit F\nRed\n\nPlayer P\nWhite\n\n=======\nLEGEND\n=======\n\n=========\nSOUNDS\n=========\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Fruit\n\n======\nRULES\n======\n\nright [ > Player | ... | Fruit ] -> [ | ... | Player ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\nF.P.....FF\n", [3, 1, 1], "background fruit:0,background:1,1,1,1,1,background player:2,1,1,0,\n", 0]
	],
	[
		"basic unlimited object/layer support",
		[{file:"test_scripts/basic_unlimited_object_layer_support.txt"}, [1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1], "background:0,0,0,0,0,0,0,0,0,0,background player:1,0,\n0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,\n", 0]
	],
	[
		"many layers movement bug",
		[{file:"test_scripts/many_layers_movement_bug.txt"}, [1], "background player:0,background:1,\n", 0]
	],
	[
		"random movement determinism test",
		[{file:"test_scripts/random_movement_determinism_test.txt"}, [3], "background wall:0,0,0,0,0,0,0,0,\nbackground:1,1,1,background crate:2,1,0,0,1,\n1,1,1,1,0,0,1,2,\n1,1,1,0,0,1,1,background player:3,\n1,1,0,0,1,1,1,1,\n2,0,0,1,2,1,1,1,\n0,0,0,0,0,0,0,0,\n", 0, "1397237766414.352"]
	],
	[
		"random instances of properties",
		[{file:"test_scripts/random_instances_of_properties.txt"}, [3], "background crate1:0,0,background crate2:1,0,0,0,1,0,\n1,1,1,1,1,1,0,1,\n0,0,0,0,1,0,1,1,\nbackground player:2,0,1,1,1,0,0,0,\n1,1,1,1,0,0,1,0,\n0,1,1,0,1,1,0,1,\n1,1,0,0,1,0,1,0,\n", 0, "1397239729215.862"]
	],
	[
		"sticky candy puzzle saga",
		[{file:"test_scripts/sticky_candy_puzzle_saga.txt"}, [2, 3, 3, 0, 3, 2, 3, 3, 2, 1], {file:"test_endstates/sticky_candy_puzzle_saga.txt"}, 4, "1397263843369.0808"]
	],
	[
		"modality",
		[{file:"test_scripts/modality.txt"}, [2, 2, 0, 3, 3, 2, 2, 2, 1, 1, 0, 1, 1, 0, 3, 3, 3, 1, 1, 1, 2, 3, 3, 3, 3, 0, 0, 0, 0], "background black:0,0,0,0,background grey:1,background white:2,\n0,2,2,0,0,2,\n0,2,2,background target white:3,0,2,\n0,0,2,0,0,2,\nbackground black crate:4,background player white:5,2,2,2,2,\n0,0,0,2,2,2,\n", 2, "1397263919028.1353"]
	],
	[
		"MazezaM",
		[{file:"test_scripts/mazezam.txt"}, [3, 3, 3, 2, 3, 3, 0, 1, 3, 1, 2, 3, 3, 3, 1, 1, 1, 0, 3, 3, 1, 2, 0, 3, 2, 1, 1, 0, 0, 0, 0, 3, 3, 3, 2, 1], "background wall:0,0,0,0,background:1,0,0,0,0,0,0,1,\n0,0,0,0,0,0,background gate:2,0,0,0,1,background lcrate:3,\n1,1,1,0,0,1,background rcrate:4,3,acrate background:5,1,0,0,\n1,background lplayer:6,4,1,1,0,0,1,1,1,3,1,\n0,0,5,5,5,background icrate:7,1,0,0,1,1,1,\n4,5,0,0,0,background exit:8,0,0,0,0,0,0,\n1,0,0,0,0,0,0,background target:9,0,0,0,0,\n", 25, "1397263965449.9036"]
	],
	[
		"Saga of the Candy Scroll",
		[{file:"test_scripts/saga_of_the_candy_scroll.txt"}, [0, 4, 3, 3, 4, 1, 1, 1, "restart", "restart", 0, 4, 2, 2, 1, 2, 2, 2, 2, 2, 4, 3, 2, 2, 3, 3, 4, 3, 3, 3, 3, 0, 0, 0, 4, 2, 2, 2, 1, 1, 4], {file:"test_endstates/saga_of_the_candy_scroll.txt"}, 70, "1397264072628.9756"]
	],
	[
		"Pants, Shirt, Cap",
		[{file:"test_scripts/pants_shirt_cap.txt"}, [2, 3, 3, 3, 1, 1, 1, 0, 2, "restart", 2, 2, 2, 3, 2, 0, 0, 0, 2, 2, 1, 0, 0, 3, 3, 3, 1, 1, 1, 2, 2, 2, 1, 3, 0, 0, 0, 3, 3, 3, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 4, 4, 1, 1, 1, 1], {file:"test_endstates/pants_shirt_cap.txt"}, 25, "1397264366021.823"]
	],
	[
		"Gobble Rush",
		[{file:"test_scripts/gobble_rush.txt"}, [1, 1, 1, "restart", 1, 3, 2, 1, 0, 0, 1, 1, 2, 0, 1, 2, 3, 2, 0, 1, 2, 0], "background:0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,1,1,1,1,0,0,1,\n0,background player:2,alert background enemy moveu:3,1,1,0,1,background enemy:4,0,0,background lettuce:5,\n1,1,1,0,1,0,5,background exit:6,1,1,0,\n0,0,5,1,1,1,0,0,0,1,1,\n0,1,1,1,1,1,0,0,0,0,0,\n0,0,0,0,0,background level0:7,7,7,background level4:8,7,0,\n", 16, "1397264727401.1748"]
	],
	[
		"Slidings",
		[{file:"test_scripts/slidings.txt"}, [1, 2, 2, 4, 2, 3, 4, 4, 2, 3], "background wall:0,0,0,0,0,0,0,background count:1,\n0,0,0,0,0,0,1,background:2,\n2,2,2,2,0,1,2,2,\n2,2,2,0,background countend:3,2,2,background target:4,\n2,background player:5,0,0,2,2,2,2,\nbackground balla:6,0,0,2,2,2,2,background ballb:7,\n0,0,0,0,0,0,0,background ballc overwall:8,\n", 17, "1397264922711.8103"]
	],	
	[
		"Flying Kick",
		[{file:"test_scripts/flying_kick.txt"}, [0, 1, 1, 0, 1, 0, 3], {file:"test_endstates/flying_kick.txt"}, 15, "1397265077451.8643"]
	],
	[
		"Cratopia",
		[{file:"test_scripts/cratopia.txt"}, [1, 1, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 0], {file:"test_endstates/cratopia.txt"}, 5, "1397265209690.2634"]
	],
	[
		"Heroes of Sokoban 3",
		[{file:"test_scripts/heroes_of_sokoban_3.txt"}, [1, 4, 1, 3, 1, 0, 2, 3, 4, 2, 3, 4, 0, 4, 1, 4, 2, 0, 4, 0, 4, 4, 1, 4, 2, 2, 4, 2, 3, 4, 2, 0, 1, 4, 1, 0, 4, 3, 4, 2, 4, 3, 4, 3, 4, 0], "background wall:0,0,0,0,0,0,0,0,\nbackground crate:1,0,background:2,0,1,0,0,background goal:3,\n0,2,0,3,0,0,background yellowswitch:4,2,\nbackground crate note:5,2,2,0,0,0,0,background yellowdoor:6,\n0,0,0,0,0,0,background bard:7,2,\n0,0,0,0,0,0,background note swizard:8,0,\n0,0,0,0,0,0,0,0,\n", 12, "1397265277461.7083"]
	],
	[
		"Manic Ammo",
		[{file:"test_scripts/manic_ammo.txt"}, [1, 1, 2, 2, 3, 4, 0, 0, 0, 0, 4, 2, 2, 2, 2, 4, 0, 3, 3, 3, 3, 3, 3, 4, 2, 4, 1, 1, "restart", "restart", 1, 0, 1, 2, 2, 2, 3, 4, 0, 0, 0, 0, 4, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 0, 3, 3, "restart", "restart", 3, 3, 2, 3, 2, 1, 4, 4, 2, 1, 2, 2, 1, 0, 1, 1, 2, 4, 0, 0, 3, 3, 2, 2, 3, 4, 4, 4, "restart", "restart", 2, 2, 3, 4, 4, 0, 0, 3, 3, 4, 3, 4, 1, 1, 4, 1, 2, 2, 3, 4, 4, 1, 0, 1, 0, 0, 4, 4, 2, 2, 4, 2, 3, 3, 3, 0, 3, 3, 2, 4, 1, 1, 0, 1, 2, 1, 1, 2, 4, 0, "undo", "undo", "undo", "undo", "undo", "undo", "undo", 1, 1, 2, 3, 0, 3, 2, 1, 1, 1, 2, 4, 0], "background wall:0,0,0,0,0,0,0,0,background:1,\n1,1,background player:2,1,0,0,1,1,1,\n1,background gune:3,0,0,1,1,1,1,1,\n0,0,1,1,1,1,1,0,0,\n1,1,1,1,1,0,0,1,1,\n1,1,background target:4,0,0,1,1,1,1,\n1,0,0,0,0,0,0,0,0,\n", 16, "1397265361355.714"]
	],
	[
		"Tunnel Rat",
		[{file:"test_scripts/tunnel_rat.txt"}, [0, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 3, 0, 3, 0, 2, 0, 3, 3, 3, 0, 1, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 2, 0, 1, 1, 1, 0, 3, 0, 3, 3, 3, 3, 0, 3, 3, 4, 0, 3, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 1, 1, 1, 1, 0, 2, 0, 3], {file:"test_endstates/tunnel_rat.txt"}, 1, "1398622969208.197"]
	],
	[
		"Cute Train",
		[{file:"test_scripts/cute_train.txt"}, [3, 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 4, 4, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 0, 4, 4, 1, 1, 2, 2, 4, 0, 0, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], {file:"test_endstates/cute_train.txt"}, 1, "1398623070351.8035"]
	],
	[
		"sokobond demake",
		[{file:"test_scripts/sokobond_demake.txt"}, [2, 2, 1, 2, 2, 3], "background wall:0,0,0,0,0,0,0,0,0,\n0,0,0,0,background:1,0,0,0,0,\n0,0,0,1,1,1,0,0,0,\n0,0,1,1,1,background bondr hydrogen:2,1,0,0,\n0,1,1,1,1,1electron background bondlrd carbon player:3,background bondu hydrogen:4,1,0,\n0,0,1,1,1,background bondl hydrogen:5,1,0,0,\n0,0,0,1,1,1electron background hydrogen:6,0,0,0,\n0,0,0,0,1,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n", 14, "1397265775629.195"]
	],
	[
		"castlemouse",
		[{file:"test_scripts/castlemouse.txt"}, [3, 1, "restart", "restart", 0, 3, 2, 1, 0, 0, 1, 2, 1, 2], {file:"test_endstates/castlemouse.txt"}, 11, "1397265815786.2612"]
	],
	[
		"atlas shrank",
		[{file:"test_scripts/atlas_shrank.txt"}, [3, 3, 3, 3, 4, 1, 1, 4, 3, 1, 1, 3, 3, 3, 3, 0, 3, 1, 1, 1, 1, 1, 1, 3], {file:"test_endstates/atlas_shrank.txt"}, 10, "1397265862607.908"]
	],
	[
		"ponies",
		[{file:"test_scripts/ponies.txt"}, [0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1], {file:"test_endstates/ponies.txt"}, 21, "1397265987570.09"]
	],
	[
		"take heart lass",
		[{file:"test_scripts/take_heart_lass.txt"}, [0, 1, 2, 3, 3, 3, 1], {file:"test_endstates/take_heart_lass.txt"}, 15, "1397266037416.4868"]
	],
	[
		"tiny treasure hunt",
		[{file:"test_scripts/tiny_treasure_hunt.txt"}, [1, 1], {file:"test_endstates/tiny_treasure_hunt.txt"}, 1, "1398623149061.288"]
	],
	[
		"heroes of sokoban 2",
		[{file:"test_scripts/hos2.txt"}, [2, 2, 0, 4, 2, 2, 2, 4, 2, 1, 1, 2, 2, 3, 2, 2], {file:"test_endstates/heroes_of_sokoban_2.txt"}, 27, "1397266289444.3076"]
	],
	[
		"dang I'm huge",
		[{file:"test_scripts/damn_i_m_huge.txt"}, [1, 0, 0, 1, 1, 2, 2, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 1, 1, 2, 2], {file:"test_endstates/dang_i_m_huge.txt"}, 13, "1397266323534.4998"]
	],
	[
		"legend of zokoban",
		[{file:"test_scripts/legend_of_zokoban.txt"}, [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 0, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 2, 0, 3, 1, 2, 2, 0, 3, 3, 3, 3, 3, 3, 3, 2, 1, 2, 3, 3, 2, 3, 0, 1, 0, 3, 3, 3, 3, 0, 3, 3, 1, 1, 2, 3, 2, 0, 0, 3, 2, 1, 2, 3, 3, 1, 0, 0, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 0, 1, 2, 3, 0, 3, 0, 3, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 0, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3], {file:"test_endstates/legend_of_zokoban.txt"}, 0, "1417791854844.9397"]
	],
	[
		"ebony & ivory",
		[{file:"test_scripts/ebony_ivory.txt"}, [3, 0, 1, 3, 0, 1, 0, 1, 2, 2, 2, 3, 3, 2, 1, 1], "background gray:0,0,0,0,0,0,0,0,\nbackground white:1,1,1,1,background playerred white:2,0,0,1,\n1,1,1,1,0,0,background black:3,1,\n1,1,1,0,0,3,3,3,\nbackground black playerred:4,1,0,0,3,3,3,3,\n3,0,0,3,3,3,3,3,\n0,0,0,0,0,0,0,0,\n", 0, "1397266474258.7976"]
	],
	[
		"explod",
		[{file:"test_scripts/explod.txt"}, [1, 4, 3, 3, 2, 2, 0, 2, 0], "background:0,0,0,background goal:1,background wall:2,2,2,\n2,0,0,0,0,0,background litbomb3:3,\nbackground unlitbomb3:4,4,0,0,3,4,background water:5,\n5,5,5,0,background liveplayer:6,0,0,\n", 9, "1397266671117.456"]
	],
	[
		"dungeon janitor",
		[{file:"test_scripts/dungeon_janitor.txt"}, [0, 0, 0, 0, 0, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 3, 3, 3, 2, 3, 2, 2, 3, 3, 0, 3, 3, 3, 3, 2, 3, 2, 3, 3, 0, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 2, 2, 2, 1, 2, 2, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1], {file:"test_endstates/dungeon_janitor.txt"}, 6, "1397266724723.82"]
	],
	[
		"icycrates",
		[{file:"test_scripts/icycrates.txt"}, [3, 2, 1, 2, 0], {file:"test_endstates/icycrates.txt"}, 7, "1397266762189.0361"]
	],
	[
		"armageddon",
		[{file:"test_scripts/armageddon.txt"}, [3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], {file:"test_endstates/armageddon.txt"}, 3, "1397266804318.573"]
	],
	[
		"love and pieces",
		[{file:"test_scripts/love_and_pieces.txt"}, [0, 3, 3, 3, 0, 3, 3, 3, 3, 2, 2, 2, 2, 1], "background wall:0,0,0,0,0,0,0,0,0,0,0,background:1,\n1,1,1,1,1,1,1,0,0,1,1,1,\n1,1,background grayblock:2,1,1,0,0,1,1,0,1,1,\n0,2,1,0,0,1,1,1,1,1,1,1,\n1,0,0,1,1,1,1,1,1,1,1,0,\n0,1,1,1,1,1,1,1,1,0,0,1,\n1,1,1,1,1,background player:3,1,0,0,1,1,0,\n1,3,0,3,1,0,0,1,1,1,1,3,\n3,3,1,0,0,1,1,1,1,1,1,1,\n1,0,0,0,0,0,0,0,0,0,0,0,\n", 3, "1397266871664.369"]
	],
	[
		"it dies in the light",
		[{file:"test_scripts/it_dies_in_the_light.txt"}, [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 3, 3, 3, 3, 3, 2, 2, 1, 1, 2, 1, 1, 2], {file:"test_endstates/it_dies_in_the_light.txt"}, 2, "1397266942631.7705"]
	],
	[
		"limerick",
		[{file:"test_scripts/limerick.txt"}, [0, 3, 3, 3, 3, 0, 3, 0, 0, 3, 0, 0, 1, 0, 0, 3, 3, 3], {file:"test_endstates/limerick.txt"}, 19, "1397269808625.158"]
	],
	[
		"2D whale world",
		[{file:"test_scripts/2d_whale_world.txt"}, [3, 1, 0, 0, 2, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1], {file:"test_endstates/2d_whale_world.txt"}, 13, "1397269841718.1611"]
	],
	[
		"Push",
		[{file:"test_scripts/push.txt"}, [0, 0, 3, 3, 3, 2, 0, 3, 3, 3, 3, 0, 0, 1, 1, 1, 3, 2, 1, "undo", 3, 0, 0, 3, 2], "background wall:0,0,0,0,0,0,0,0,0,0,0,\n0,background rock:1,background:2,background weakground:3,2,2,1,2,2,2,0,\n0,1,1,1,1,3,1,2,1,1,0,\n0,1,background pit:4,1,1,2,1,2,1,2,0,\n0,1,2,1,1,3,1,2,2,1,0,\n0,background idol:5,2,1,1,4,1,2,1,2,0,\n0,1,1,1,1,2,1,2,1,2,0,\n0,1,1,1,1,2,1,4,1,2,0,\n0,1,1,1,1,2,background pitplayer:6,2,1,2,0,\n0,1,1,1,1,1,1,1,1,2,0,\n0,0,0,0,0,0,0,0,0,0,0,\n", 5, "1397270241789.4006"]
	],
	[
		"Kettle",
		[{file:"test_scripts/kettle.txt"}, [1, 1, 3, 2, 0], {file:"test_endstates/kettle.txt"}, 22, "1397303719628.0938"]
	],
	[
		"Bouncers",
		[{file:"test_scripts/bouncers.txt"}, [0, 3, 3, 2, 2, 2, 2, 3, 3, 4, 3, 0, 1, 1, 3, 4, 3, 1, 1, 1, 1, 3, "restart", 2, 2, 2, 2, 1, 1, 2, 4, "restart", 0, 2, 2, 3, 3, 3, 2, 3, 4, 3, 4, 0, 0, 0, 4, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 4], {file:"test_endstates/bouncers.txt"}, 3, "1397303951623.1404"]
	],
	[
		"Bouncers 2",
		[{file:"test_scripts/bouncers.txt"}, [3, 3, 3, 3, 1, 1, 1, 1, 4, 3, 3, 1, 4, 2, 3, 3, 3, "restart", 3, 3, 3, 1, 4, 1, 4, 1, 4, 1, 4, 2, 2, 2, 2, 2, 4, 0, 3, 3, 2, 2, 1, 0, 3, "restart", 3, 3, 4, 1, 1, 3, 4, 1, 4, 1, 4, 2, 2, 2, 2, 4, 2, 0, 2, 4, 0, 0, 3, 3, 2, 1, 0, 3, 2, "undo", 0, 0, 0, 0, 3, 3, 0, 3, 4, 1, 4, "restart", "restart", 3, 3, 4, 1, 4, 1, 4, 2, 2, 2, 1, 2, 2, 4, "undo", 4, "undo", 0, 0, 0, 0, 0, 3, 3, 4, 3, 4, "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", 1, 1, 2, 2, 2, 2, 2, 4, "undo", "undo", 2, 4, "undo", "undo", 0, 0, 0, 0, 3, 4, 2, 2, 2, 2, 1, 2, "undo", 2, 4], {file:"test_endstates/bouncers_2.txt"}, 7, "1397304013801.759"]
	],
	[
		"Sok7",
		[{file:"test_scripts/sok7.txt"}, [0, 3, 3, 2, 3, 0, 3, 0, 1, 2, "undo", 2, 3, 3, 3, 3, "undo", "undo", 1, 3, 3, 3, 1, 1, 2, 1, 0, 0, 3, 2, 1, 1, 2, 1, 0, 1, 1, 1, 0, 0, 3, 3, 3, 2, 3, 3, 0, 1, 2, "undo", 2, 1, 0, 3, 0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 1, 1, 3, 3, 2, 2, 1, 0, 3, 0, 1, 3, 3, 0, 0, 0, 1, 2, 2, 3, 2, 1, 1, 3, 3, 3, 0, 3, 3, 2, 1, 1, 1, 1, 3, 0, 0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 0, 3, 3, 3, "undo", 2, 3, 3, 3, 3, 0, 0, 1, 1, 3, 3, 3, 3, 3, 2, 1, 1, 3, 3, 3, 2, 1, 1, 2, 1, 0, 3, 1, 1], {file:"test_endstates/sok7.txt"}, 0, "1397736919163.304"]
	],
	[
		"Closure",
		[{file:"test_scripts/closure.txt"}, [3, 4, 3, 0, 3, 4, 1, 3, 3, 3, 0, 3, 4, 0, 0, 3, 0, 1, 1, 3, 0, 3, 0, 3, 3, 4, 0, 3, 4, 3, 0, 1, 1, 3, 3, 3, 3, 0, 3, 4, 3, 4, 0, 1, 0, 3, 0, 2], {file:"test_endstates/closure.txt"}, 3, "1397314168422.1912"]
	],
	[
		"Color Chained",
		[{file:"test_scripts/color_chained.txt"}, [1, 1, 2, 3, 2, 3, 2, 1, 2], {file:"test_endstates/color_chained.txt"}, 10, "1397314339663.9756"]
	],
	[
		"Drop Swap",
		[{file:"test_scripts/drop_swap.txt"}, [0, 1, 1, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0], {file:"test_endstates/drop_swap.txt"}, 3, "1397737011377.065"]
	],
	[
		"Drop Swap 2",
		[{file:"test_scripts/drop_swap.txt"}, [0, 1, 2, 1, 3, 0, 1, 2, 3, 0, 1], {file:"test_endstates/drop_swap_2.txt"}, 5, "1397737044719.822"]
	],
	[
		"Drop Swap 3",
		[{file:"test_scripts/drop_swap.txt"}, ["restart", 3, 0, 0, 3, 3, 1, 0, 1, 2, 3, 0, 0, 1, 2, 2, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 1, 0, 3, 2], {file:"test_endstates/drop_swap_3.txt"}, 8, "1397737071059.7979"]
	],
	[
		"Threes",
		[{file:"test_scripts/threes.txt"}, [1, 0, 0, 1, 0, 1], {file:"test_endstates/threes.txt"}, 0, "1397316107166.7544"]
	],
	[
		"Smother",
		[{file:"test_scripts/smother.txt"}, [2, 2, 3, 3, 3, 3, 2, 3], {file:"test_endstates/smother.txt"}, 8, "1397318526578.8389"]
	],
	[
		"again + message combo",
		["========\nOBJECTS\n========\n\nBackground\nBlack\n\nTrigger\nRed\n.....\n.000.\n.0.0.\n.000.\n.....\n\nPlayer\nGreen\n00000\n0...0\n0.0.0\n0...0\n00000\n\n=======\nLEGEND\n=======\n\n. = Background\nX = Trigger\nP = Player\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTrigger\nPlayer\n\n======\nRULES\n======\n\nlate [ Player Trigger ] -> [ Trigger ] again message Click!\n\n==============\nWINCONDITIONS\n==============\n\nno Trigger\n\n=======\nLEVELS\n=======\n\nmessage Level 1\n\n.....\n.....\n.P.X.\n.....\n.....\n\nmessage Level 2\n\n.....\n..P..\n.....\n..X..\n.....\n\n", [3, 3], "background:0,0,0,0,0,\n0,0,0,0,0,\n0,0,0,0,0,\n0,0,background trigger:1,0,0,\n0,0,0,0,0,\n", 1, "1397321235736.0955"]
	],
	[
		"chaos wizard",
		[{file:"test_scripts/chaos_wizard.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", 1, "tick", 3, "tick", "tick", 2, "tick", 2, "tick", 2, "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "restart", "tick", "tick", 3, "tick", 3, 2, "tick", 2, 2, "tick", 2, "tick", 2, "tick", 2, 3, "tick", 3, "tick", 3, 0, "tick", 0, "tick", 0, 1, 1, "tick", 1, "tick", 1, "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/chaos_wizard.txt"}, 3, "1397323474349.4995"]
	],
	[
		"50 objects",
		[{file:"test_scripts/50_objects.txt"}, [4, 1, 3, 0, 1, 2, 2, 3, 3, 0, 3, 0, 1, 1, 2, 2, 2, 1, 0], {file:"test_endstates/50_objects.txt"}, 0, "1397342665552.5623"]
	],
	[
		"100 objects",
		[{file:"test_scripts/100_objects.txt"}, [4, 1, 2, 1, 1], "background river:0,0,0,0,0,0,background o9:1,background o2:2,background player:3,\n0,0,background o27:4,background o37:5,background:6,0,0,background n0:7,6,\n6,background o49 river:8,0,background n22:9,6,background n14:10,0,0,5,\nbackground n37:11,background n27:12,0,0,background n18:13,background n8:14,background o3:15,0,0,\nbackground n48:16,background o31:17,background n24:18,0,0,0,0,0,0,\n", 0, "1397344224335.4502"]
	],
	[
		"200 objects",
		[{file:"test_scripts/200_objects.txt"}, [4, 1, 0, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 0, 3, 0, 1, 4, 3, 3, 3, 2, 2, 3, 3, 0, 0, 3], "background river:0,0,0,0,0,0,background:1,background m36:2,background l38:3,\n0,0,1,background n46:4,background o13:5,0,0,1,background m17:6,\nbackground l32:7,0,0,1,1,1,0,0,background l34:8,\nbackground o38:9,1,0,0,1,1,1,0,0,\nbackground player:10,background n21:11,background m18:12,0,0,0,0,0,0,\n", 0, "1397343088681.431"]
	],
	[
		"misc many objects",
		[{file:"test_scripts/misc_many_objects.txt"}, [4, 1, 0, 3, 3, 3, 2, 2], "background river:0,0,0,0,0,0,background o10:1,background o8:2,background o47:3,\n0,0,background o1:4,background o45:5,background o35:6,0,0,background:7,7,\nbackground o16:8,0,0,7,7,2,0,0,7,\nbackground o23:9,5,0,0,7,7,background player:10,0,0,\nbackground o11:11,background o15:12,background o12:13,0,0,0,0,0,0,\n", 0, "1397343233919.3174"]
	],
	[
		"50 layers",
		[{file:"test_scripts/50_layers.txt"}, [4, 1, 0, 3, 3, 3, 2, 2], "background river:0,0,0,0,0,0,background o21:1,background o42:2,background o17:3,\n0,0,background o36:4,background o39:5,background o28:6,0,0,background:7,7,\nbackground o38:8,0,0,7,7,background o32:9,0,0,7,\nbackground o19:10,background o22:11,0,0,7,7,background player:12,0,0,\nbackground o14:13,background o8:14,background o34:15,0,0,0,0,0,0,\n", 0, "1397343421304.0276"]
	],
	[
		"50 layers 2",
		[{file:"test_scripts/50_layers_2.txt"}, [1, 4, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3], "background river:0,0,0,0,0,0,background o9:1,background o47:2,background o29:3,\n0,0,background o3:4,background o23:5,background o8:6,0,0,background o27:7,background player:8,\nbackground o18:9,0,0,5,background o33:10,background o21:11,0,0,background o12:12,\n5,background o24:13,0,0,background o16:14,7,background o32:15,0,0,\nbackground o22:16,14,background o49:17,0,0,0,0,0,0,\n", 0, "1397343495880.7278"]
	],
	[
		"netrunner test",
		[{file:"test_scripts/netrunner_test.txt"}, [0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, "tick", 3, 3, 0, 4, 4, "tick", 2, 2, "tick", 0, 0, 2, 0, 4, "tick", 3, 4, "tick"], {file:"test_endstates/netrunner_test.txt"}, 1, "1397348749712.9993"]
	],
	[
		"wb + gems test",
		[{file:"test_scripts/wb_gems_test.txt"}, [3, 3, 2, 0, 3, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 0, 2, 2, 1, 1, 1, 2, 2], {file:"test_endstates/wb_gems_test.txt"}, 16, "1397352258217.8733"]
	],
	[
		"10 layers 50 objects",
		[{file:"test_scripts/10_layers_50_objects.txt"}, [1, 0, 3, 3, 2, 1, 1, 0, 3, 3, 2, 1, 3, 0, 1, 0, 3, 3, 3, 1, 1, 2, 3, 2, 3, 0, 1, 1, 1, 2, 1, 0, 0, 3, 3, 3, 3, 1, 1, 1, 2, 1, 2, 3, 3, 3, 2, 3, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 3, 3, 0, 2, 3, 3, 3, 2, 3, 0, 1, 1, 1, 2, 3, 3, 3, 0, 0, 1, 1, 2, 2, 3, 2, 3, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 1, 1, 3, 3, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 0, 0, 1, 1, 1, 2, 3, 1, 0, 3, 3, 3, 1, 2, 2, 0, 1, 0, 3, 1, 1, 0, 3, 3, 1, 2, 2, 2, 3, 3, 0, 0, 1, 1, 0, 3, 3, 1, 2, 2, 3, 0, 1, 1, 2, 3, 3, 0, 1, 0, 3, 1, 2, 2, 2, 3, 3, 0, 0, 1, 1, 3, 2, 1], {file:"test_endstates/10_layers_50_objects.txt"}, 1, "1397353520415.2493"]
	],
	[
		"10 layers, 50 objects 2",
		[{file:"test_scripts/10_layers_50_objects_2.txt"}, [3, 3], "background:0,0,0,0,\n0,0,0,0,\n0,0,background l9_4 player:1,0,\n", 0, "1397387511600.6943"]
	],
	[
		"throttle movement",
		[{file:"test_scripts/throttle_movement.txt"}, [0, 1, 1, 1, 1], "background wall:0,0,0,0,0,background:1,0,1,\n1,1,0,1,0,background player:2,background target:3,1,\n0,1,0,1,background crate target:4,1,0,1,\n0,1,4,1,0,0,0,1,\nbackground crate:5,1,1,0,0,1,1,1,\n1,0,0,0,0,0,0,0,\n", 7, "1397558503752.5374"]
	],
	[
		"robotic arm",
		[{file:"test_scripts/robotic_arm.txt"}, [0, 0, 3, 1, 2, 2, 4, 3, 0, 0, 0, 1, 0, 0, 3, 3, 3, 2, 0, 1, 2, 0, 1, 1, 2, 3, 0, 3, 2, 4, 1, 2, 2, 3, 1, 0, 0, 3, 2, 2, 1, 2, 2, 3, 0, 0, 1, 0, 0, 3, 2, 2, 4, 1, 2, 2, 3, 0, 0, 0, 1, 0, 0, 3, 3, 2, 1, 2, 2, 1, 3, 0, 0, 0, 1, 1, 2, 3, 0, 3, 2], {file:"test_endstates/robotic_arm.txt"}, 1, "1397688118644.9731"]
	],
	[
		"circuit game",
		[{file:"test_scripts/circuit_game.txt"}, [0, 0, 0, 1, 2, 2, 3, 2, 2, 0, 0, 1, 3, 3, 2, 2, 1, 3, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 3, 0, 3, 3, 3, 2, 2, 2, 3, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 3, 2, 2, 3, 1, "undo", 0, 3, 0, 3, 3, 3, 2, 1, 1, 0, 1, 1, 2, 1, 2, 3, 1, 0, 3, 3, 0, 3, 2, 2], "background:0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,\n0,background cable_ul:1,0,0,0,0,background red_on:2,\nbackground battery_v light light_down light_up:3,background cable_ur light light_right light_up:4,0,0,0,background player:5,background green_on:6,\nbackground cable_ulr light light_left light_right light_up:7,background blue_off:8,0,0,0,0,0,\n0,0,0,0,0,0,0,\n", 5, "1397688202960.5383"]
	],
	[
		"multiple patterns not checking for modifications",
		["title Simple Block Pushing Game\nauthor increpare\nhomepage www.puzzlescript.net\nnorepeat_action\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\n\nTarget\nDarkBlue\n.....\n.000.\n.0.0.\n.000.\n.....\n\nWall\nBROWN DARKBROWN\n00010\n11111\n01000\n11111\n00010\n\nPlayer\nBlack Orange White Blue\n.000.\n.111.\n22222\n.333.\n.3.3.\n\nCrate\nOrange Yellow\n00000\n0...0\n0...0\n0...0\n00000\n\n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\n@ = Crate and Target\nO = Target\n\n\n=======\nSOUNDS\n=======\n\nCrate MOVE 36772507\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES\n======\n\n[ action Player ] [ Crate ] -> [ Player ] [ ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n\n**P\n\n", [4], "background:0,background crate:1,background player:2,\n", 0, "1397736831873.7236"]
	],
	[
		"2048",
		[{file:"test_scripts/2048.txt"}, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], "6 background:0,5 background:1,3 background:2,1 background:3,\n1,2,2 background:4,background:5,\n4 background:6,3,3,5,\n4,5,5,5,\n", 0, "1397902282955.9907"]
	],
	[
		"bug #186 - 'no property' on the RHS should clear objects",
		[{file:"test_scripts/bug_186_no_property_on_the_rhs_should_clear_objects.txt"}, [1, 1, 2, 2, 3, 3, 2, 2], "background wall:0,0,0,background:1,1,1,\n0,0,1,1,background target:2,1,\n1,0,0,2,1,1,\n1,1,background player:3,0,0,0,\n1,background crate:4,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1398621035085.1448"]
	],
	[
		"Make synonyms of properties work. #215",
		["========\nOBJECTS\n========\n\nMoveUp\nwhite\n\nBackground\nblack\n\nHarpoonU\ngrey\n\nHarpoonD\nwhite\n\nPlayer\nyellow\n\n=======\nLEGEND\n=======\n\n. = Background\n\n(this double indirection seems to be the problem?)\nHarpoon = HarpoonU or HarpoonD \nMoveable = Harpoon\n\nP = Player\nU = HarpoonU\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nMoveUp\nBackground\nPlayer Harpoon\n\n======\nRULES\n======\n\n[ MoveUp Moveable ] -> [ up MoveUp up Moveable ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nPU\n\n", [3], "background player:0,background harpoonu:1,\n", 0, "1409382951213.9463"]
	],
	[
		"Make synonyms of properties work. #230",
		[{file:"test_scripts/make_synonyms_of_properties_work_230.txt"}, [1, 1, 1], "background wall:0,0,0,0,0,0,0,0,background:1,adultbunny background:2,\n1,1,1,0,0,1,1,1,1,1,\n0,0,0,1,1,1,0,0,1,0,\n1,background player:3,1,0,1,1,0,2,1,1,\n0,1,1,0,1,1,1,0,1,1,\n0,1,1,1,0,1,1,0,1,1,\n1,0,1,1,0,0,0,0,0,1,\n", 0, "1409383029666.0354"]
	],
	[
		"Make synonyms of properties work. #241",
		[{file:"test_scripts/make_synonyms_of_properties_work_241.txt"}, [3, 3, 3, 3, 3], "background wall:0,0,0,0,0,0,0,0,0,\n0,background:1,1,1,1,1,0,background inv:2,0,\n0,1,1,1,1,1,0,2,0,\n0,1,1,1,1,1,0,2,0,\n0,1,1,1,1,1,0,2,0,\n0,1,1,1,1,1,0,2,0,\n0,1,1,background player:3,1,1,0,2,0,\n0,1,1,background exit:4,1,1,0,2,0,\n0,0,0,0,0,0,0,0,0,\n", 0, "1409383068970.6855"]
	],
	[
		"Make synonyms of properties work. #243",
		["========\nOBJECTS\n========\n\nBackground\nBlack\n\nPlayer1\nBlack\n\nPlayer2\nBlack\n\n=======\nLEGEND\n=======\n\nAnyPlayer = Player1 or Player2\n\n((works:)\nPlayer = Player1 or Player2\n)\n\n(fails:)\nPlayer = AnyPlayer\n\n. = Background\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nAnyPlayer\n\n======\nRULES\n======\n\n[ Player ] -> [ ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n.\n", [3], "background:0,\n", 0, "1409383122377.9321"]
	],
	[
		"Make synonyms of properties work. #245",
		[{file:"test_scripts/make_synonyms_of_properties_work_245.txt"}, ["tick", "tick", "tick", "tick", "tick"], "platform:0,0,0,0,0,platform player:1,0,0,0,0,0,bluecrate platform:2,0,\n0,0,0,0,0,0,0,0,0,0,2,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,hunterr platform:3,0,0,0,0,\n0,0,2,0,0,0,0,0,0,0,0,0,0,\n0,0,platform redcrate:4,0,0,0,hole1 platform:5,5,5,hole2 platform:6,5,5,5,\n", 2, "1409383166222.726"]
	],
	[
		"Failed rigid groups shouldn't block late rule execution. #254",
		[{file:"test_scripts/failed_rigid_groups_shouldn_t_block_late_rule_execution_254.txt"}, [3, 3], "background wall:0,0,0,0,0,0,0,0,background:1,\n1,1,1,1,0,0,1,1,1,\n1,1,0,0,1,1,1,1,1,\n0,0,1,background player:2,1,1,1,0,0,\n1,1,1,1,1,0,0,1,1,\n1,1,1,0,0,1,1,1,1,\n1,0,0,0,0,0,0,0,0,\n", 0, "1409552619480.2417"]
	],
	[
		"Rigidbody fix bug #246",
		[{file:"test_scripts/rigidbody_fix_bug_246.txt"}, [0, 0, "restart", 0, 0, "restart", 0, 3, 0, 0, 0, 2, 2, 2, 2, "restart", 0, 0, "restart", 0, 0, "restart", 0, 0, 0, 0, 3, 2, 2, 2, 0, 0, 0, 3, 3, 0, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 0, 0, 1, 1, "restart", 0, 1, "restart", 1, 0, 3, 0, 2, 2, 3, 0, 0, 0, "restart", 0, 3, 0, 0, 2, 2, 2, 1, 3, 0, 1, "restart", 1, 0, 3, 3, 3, 3, 3, 2, 3, 0, 1, 0, 0, 0, 0, 3, 3], {file:"test_endstates/rigidbody_fix_bug_246.txt"}, 1, "1409552865429.6877"]
	],
	[
		"Rigidbody rules apply to group rather than individual rules.",
		[{file:"test_scripts/rigidbody_rules_apply_to_group_rather_than_individual_rules.txt"}, [1, 0, 1, 3, 2, 2, 2, 1, 0, 2, 2], "background:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,background greencrate:1,0,0,background wall:2,\n1,0,0,1,1,background player:3,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n", 0, "1409615719060.894"]
	],
	[
		"Laser movement check (#264)",
		["title Bug\n\n========\nOBJECTS\n========\n\nBackground\nDarkGrey\n\nPlayer\nGrey White\n.....\n..0..\n.010.\n..0..\n.....\n\nGun\nGrey\n..000\n..000\n000.0\n..000\n..000\n\nLaser\nRed\n.....\n.....\n00000\n.....\n.....\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nG = Gun\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nLaser\nGun Player\n\n\n======\nRULES\n======\n\n[Laser no Gun] -> []\n[] -> [laser]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n.....\n..P..\n.....\n..G..\n.....\n", [1, 2, 2], "background laser:0,0,0,0,0,\n0,0,0,background laser player:1,0,\n0,0,0,background gun laser:2,0,\n0,0,0,0,0,\n0,0,0,0,0,\n", 0, "1417791963112.0676"]
	],
	[
		"Undo test (#315)",
		[{file:"test_scripts/undo_test_315.txt"}, [2, 2, "undo", 3, 3, "undo", 2, 2, 3, "undo", "undo", "undo", 2, 2, "undo", 2, "undo", "undo", 2, 3, 3, 3, 2, 3, 1, 0, 3, 2, "undo", "undo", "undo", "undo", "undo", "undo", 3], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,\n1,0,0,background target:3,1,1,\n1,1,0,0,0,0,\nbackground player:4,background crate:5,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1465308519218.314"]
	],
	[
		"Rigid weirdness test (#369)",
		["title Rigid Clothes\nauthor ThatScar\n\nverbose_logging\ndebug\n\n========\nOBJECTS\n========\n\nBackground .\nLightGreen Green\n11111\n01111\n11101\n11111\n10111\n\nPlayer \nBlack Orange\n.000.\n.111.\n11111\n.111.\n.1.1.\n\nSumo A\nPink White\n..0..\n00000\n.000.\n.111.\n.0.0.\n\nShirt T\nWhite\n.....\n.....\n00000\n.000.\n.....\n\nPants U\nBlue\n.....\n.....\n.....\n.000.\n.0.0.\n\n=======\nLEGEND\n=======\n\nClothes = Shirt or Pants\nP = Player and Shirt and Pants\n\n=======\nSOUNDS\n=======\n\nplayer action 1232\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Sumo\nShirt\nPants\n\n======\nRULES\n======\n\n[ Parallel Player | Sumo ] -> [ Parallel Player | Parallel Sumo ]\n\nRigid [ Moving Player Clothes ] -> [ Moving Player Moving Clothes ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\nPA.\n\n", [3], "background:0,background pants player shirt:1,background sumo:2,\n", 0, "1487773403601.2908"]
	],
	[
		"Synonym confusion",
		["title Simple Block Pushing Game\nauthor increpare\nhomepage www.puzzlescript.net\n\n========\nOBJECTS\n========\n\nBackground\nRed\n\nPlayer\nOrange\n\nFloorB2\nYellow\n\nFloorB1\nGreen\n\nFloorMain\nBlue\n\n=======\nLEGEND\n=======\n\nFloorB2AndDown = FloorB2\nFloorB1AndDown = FloorB1 and FloorB2AndDown\nFloorMainAndDown = FloorMain and FloorB1AndDown\n\n. = Background\nP = Player\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nFloorB2\nFloorB1\nFloorMain, Player\n\n======\nRULES\n======\n\n==============\nWINCONDITIONS\n==============\n\nno Player\n\n=======\nLEVELS\n=======\n\n...\n..P\n", [1], "background:0,0,0,\nbackground player:1,0,0,\n", 0, "1496933845000"]
	],
	[
		"a = b and b #393",
		[{file:"test_scripts/a_b_and_b_393.txt"}, [0, 2, 2, 1, 0], "background wall:0,0,0,0,0,0,\n0,0,background:1,background crate:2,background player:3,1,\n1,0,0,background target:4,1,1,\n1,1,0,0,0,0,\n1,2,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1499256887390.9216"]
	],
	[
		"Rule direction inference test (#407)",
		[{file:"test_scripts/rule_direction_inference_test_407.txt"}, [2, 3, 2, 2, 1, 1, 0, 2, 3, 0, 2, 1, 0, "undo", 1, 0, 3, 3, 2, "undo", "undo", 2, 3, 3, 0, 1], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,1,1,\n1,0,0,1,1,1,\nbackground crate:2,1,0,0,1,1,\n1,background player:3,1,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", 1, "1503122963823.3987"]
	],
	[
		"Reserved keywords are too greedy (#419)",
		["========\nOBJECTS(cat)\n========\n\nbackground \nwhite\n\nPlayer\nblue\n\nLevelscat\nblack\n\n=======\nLEGEND\n=======\n\nP = Player\n\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS \n================\n\nbackground\nplayer\nLevelscat\n\n======\nRULES\n======\n\n\n==============\nWINCONDITIONS\n==============\n\n(there's no way to win against a floating eyeball :S )\n\n=======\nLEVELS\n=======\n\np\n\n", [3, 3, 0], "background player:0,\n", 0, "1506517029540.7014"]
	],
	[
		"Removing background tiles breaks \"no X\" wincondition (#534)",
		["title no Coin, all Player on Target\nauthor Zachary Abel\nhomepage zacharyabel.com\n\nrun_rules_on_level_start\n\n=======\nOBJECTS\n=======\n\nBackground\ngreen\n\nPlayer\nRed\n.000.\n.0.0.\n.000.\n.0...\n.0...\n\n\n======\nLEGEND\n======\n\nP = Player\n. = Background\n\n======\nSOUNDS\n======\n\n===============\nCOLLISIONLAYERS\n===============\n\nBackground\nPlayer\n\n=====\nRULES\n=====\n\n[Background] -> []\n\n=============\nWINCONDITIONS\n=============\n\nno Player\n\n======\nLEVELS\n======\n\nP.\n\n", [3, 3], ":0,player:1,\n", 0, "1612779741066.1199"]
	],
	[
		"should allow overlapping things #512",
		[{file:"test_scripts/should_allow_overlapping_things_512.txt"}, [3, 0, 3, 3], "background wall1:0,0,0,0,0,0,0,0,0,\n0,0,background:1,1,1,1,1,1,1,\n1,0,0,1,1,background target:2,1,1,1,\n1,1,0,0,1,1,1,1,1,\n1,1,1,0,0,background tuer1:3,1,1,1,\nbackground crate1:4,1,1,1,0,0,1,1,1,\n1,1,1,1,1,0,0,1,1,\n2,1,1,1,1,1,0,0,1,\n1,1,1,1,background player1:5,1,1,0,0,\n0,0,0,0,0,0,0,0,0,\n", 0, "1612782799099.4512"]
	],
	[
		"trouble parsing non-ascii in unicode #528",
		[{file:"test_scripts/trouble_parsing_non_ascii_in_unicode_528.txt"}, [3, 0, 3], "background mauer:0,0,0,0,0,0,0,\n0,0,background 大:1,background:2,2,2,2,\n2,0,0,2,2,2,2,\n2,2,0,0,2,2,2,\n2,background player:3,2,0,0,2,2,\n2,2,2,2,0,0,2,\n2,2,2,2,2,0,0,\n0,0,0,0,0,0,0,\n", 0, "1612797615322.4563"]
	],
	[
		"Can't start message with section keywords #490",
		[{file:"test_scripts/can_t_start_message_with_section_keywords_490.txt"}, [3], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,\n1,0,0,background target:3,1,1,\n1,1,0,0,0,0,\nbackground player:4,background crate:5,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 4, "1612811326612.0864"]
	],
	[
		"Bug: \"norestart\" prelude option can interfere with \"restart\" command #411",
		[{file:"test_scripts/bug_norestart_prelude_option_can_interfere_with_restart_comm.txt"}, [0, "restart", 1, 2, 2, "restart", 2, 2, 0, 3], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,\n1,0,0,background target:3,1,background player:4,\n1,1,0,0,0,0,\n1,background crate:5,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1612832538428.8394"]
	],
	[
		"undoing reset undoes two steps, not one #453",
		[{file:"test_scripts/undoing_reset_undoes_two_steps_not_one_453.txt"}, [2, 2, "restart", "undo"], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,\n1,0,0,background target:3,1,1,\n1,background player:4,0,0,0,0,\n1,background crate:5,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1612997864833.277"]
	],
	[
		"Recent update broke some games with tabs in the source code #609",
		[{file:"test_scripts/recent_update_broke_some_games_with_tabs_in_the_source_code.txt"}, [2, 2, 1, 0, 3, 0, 3, 3, 1, 1, 2, 1, 0], "background wall:0,0,0,0,0,0,\n0,0,background:1,background crate:2,background player target:3,1,\n1,0,0,background target:4,1,1,\n1,1,0,0,0,0,\n1,2,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1615587622342.6877"]
	],
	[
		"Test travelling through multiple levels",
		[{file:"test_scripts/undoing_reset_undoes_two_steps_not_one_453.txt"}, [2, 1, 0, 3, 3, 3, 2, 1, 0, 1, 1, 2, 2, 3, 0, 1, 0, 3, 0, 0, 1, 2, 3, 2, 2, 3, 3, 0, 1, 2, 1, 0, 0, 3, 2, 2, 1, 0, 0], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,1,1,\n1,0,0,1,0,background crate:2,\nbackground crate target:3,1,0,0,2,background player:4,\nbackground target:5,5,1,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", 0, "1627477832505.31"]
	],
	[
		"Win condition test \"NO X\"",
		[{file:"test_scripts/win_condition_test_no_x.txt"}, [0, 3, 3], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,1,1,\n1,0,0,1,0,background crate:2,\nbackground target:3,1,0,0,1,background player:4,\nbackground crate target:5,5,1,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", 0, "1627477994465.8933"]
	],
	[
		"Win condition test \"SOME X\"",
		[{file:"test_scripts/win_condition_test_some_x.txt"}, [3, 0, 3], "background:0,0,0,0,0,\n0,background player:1,0,0,0,\n", 0, "1627478110201.5347"]
	],
	[
		"Win condition test \"SOME X ON Y\"",
		[{file:"test_scripts/win_condition_test_some_x_on_y.txt"}, [1, 4, 3, 0, 4, 2, 3, 4, 2, 4, 1, 1, 4], "background crate:0,background player:1,background:2,2,0,\n2,2,2,2,2,\n", 0, "1627478147227.8706"]
	],
	[
		"Win condition test \"NO X ON Y\"",
		[{file:"test_scripts/win_condition_test_no_x_on_y.txt"}, [0, 1, 2, 3, 2, 2, 1, 1, 2, 3], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,1,background crate:2,\n1,0,0,1,0,2,\nbackground target:3,1,0,0,1,1,\nbackground crate target:4,3,background player:5,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", 0, "1627478173305.2952"]
	],
	[
		"Test for cancel cancelling restart #374",
		[{file:"test_scripts/test_for_cancel_cancelling_restart_374.txt"}, [2, 1, 0, 4, 4, 3, 4, 4], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,\n1,0,0,background target:3,1,1,\nbackground player:4,1,0,0,0,0,\n1,background crate:5,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1627560547946.0017"]
	],
	["Test for trigger message at same time as reset",
		[{file:"test_scripts/test_for_trigger_message_at_same_time_as_reset.txt"}, [2, 4, 1, 2, 2, 4, 0], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,\n1,0,0,background target:3,background player:4,1,\n1,1,0,0,0,0,\n1,background crate:5,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1627560815572.0037"]
	],
	["Test for trigger message at same time as cancel",
		[{file:"test_scripts/test_for_trigger_message_at_same_time_as_cancel.txt"}, [2, 2, 3], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,\n1,0,0,background target:3,1,background player:4,\n1,1,0,0,0,0,\n1,background crate:5,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1627560887718.9905"]
	],
	[
		"movement not getting correctly cleared from tile - #492",
		[{file:"test_scripts/movement_not_getting_correctly_cleared_from_tile_492.txt"}, [3, 0, 1, 1], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background player:2,background crate:3,\n1,0,0,background target:4,1,1,\n3,1,0,0,0,0,\n1,1,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1627732650514.1125"]
	],
	[
		"second test for #492 movement not getting correctly cleared from tile",
		[{file:"test_scripts/second_test_for_492_movement_not_getting_correctly_cleared_f.txt"}, [3], "background:0,0,0,0,0,0,background target:1,0,0,0,0,background crate:2,0,\n0,0,0,0,0,2,0,0,0,0,0,0,background person1:3,\n0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,background wall:4,4,4,4,4,4,4,0,0,0,\n0,0,0,1,0,0,0,0,2,0,0,0,0,\n0,0,2,0,0,0,0,0,0,background person2:5,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,0,\n", 0, "1627732858112.287"]
	],
	[
		"third test for #492 movement not getting correctly cleared from tile",
		["title Simple Block Pushing Game\nauthor Stephen Lavelle\nhomepage www.puzzlescript.net\n\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\n\nplayer p\nBlack Orange White Blue\n.000.\n.111.\n22222\n.333.\n.3.3.\n\n\nCrate\nOrange Yellow\n00000\n0...0\n0...0\n0...0\n00000\n\n\n=======\nLEGEND\n=======\n\n. = Background\n* = Crate\n\n\n=======\nSOUNDS\n=======\n\nCrate MOVE 36772507\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate\n\n======\nRULES\n======\n\n[ > Player | Crate ] -> [ > Player | > Crate ]\n\n[ > Player | ] -> [ Crate | Player ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\np*.\n\n", [3], "background crate:0,background player:1,background:2,\n", 0, "1627732919488.2104"]
	],
	[
		"fourth test for #492 movement not getting correctly cleared from tile",
		["title Simple Block Pushing Game\nauthor Stephen Lavelle\nhomepage www.puzzlescript.net\n\nverbose_logging\ndebug\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\n\nplayer p\nBlack Orange White Blue\n.000.\n.111.\n22222\n.333.\n.3.3.\n\ncrate\nred\n\n\n=======\nLEGEND\n=======\n\n. = Background\n\n\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\ncrate\n======\nRULES\n======\n\nright [> player | crate ] -> [ > player | > crate ]\nright [ > Player | ] -> [ Player | Player ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\npp.\n\n", [3], "background player:0,0,background:1,\n", 0, "1627733021968.0657"]
	],
	[
		"fifth test for #492 movement not getting correctly cleared from tile",
		["title Simple Block Pushing Game\nauthor Stephen Lavelle\nhomepage www.puzzlescript.net\n\nverbose_logging\ndebug\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\n\nplayer p\nBlack Orange White Blue\n.000.\n.111.\n22222\n.333.\n.3.3.\n\ncrate\nred\n\n\n=======\nLEGEND\n=======\n\n. = Background\n\n\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\ncrate\n======\nRULES\n======\n\nright [ > Player | ] -> [ Player | Player ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\npp.\n\n", [3], "background player:0,0,background:1,\n", 0, "1627733036654.9438"]
	],
	[
		"test for convention #2 to issue right [ > Player | ] -> [ Player | Player ]🐱‍🐉 #679 : Should right [ Player | ] -> [ Player | Player ] propagate movements? (A: yes?)\t\t",
		["title Simple Block Pushing Game\nauthor Stephen Lavelle\nhomepage www.puzzlescript.net\n\nverbose_logging\ndebug\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\n\nplayer p\nBlack Orange White Blue\n.000.\n.111.\n22222\n.333.\n.3.3.\n\ncrate\nred\n\n\n=======\nLEGEND\n=======\n\n. = Background\n\n\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\ncrate\n======\nRULES\n======\n\nrandom right [ Player | ] -> [ | Player ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n..p....\n\n", [3], "background:0,0,0,background player:1,0,0,0,\n", 0, "1627733171811.7598"]
	],
	[
		"right [ vertical Player | perpendicular Player ] -> [ perpendicular Player | ] produces error #682",
		["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nbaba = player or Background\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\nright [ up player | up player ] -> [ up player | down player ]\nright [ vertical Player | perpendicular Player ] -> [ perpendicular Player | ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...pp..\n.......\n", [2], "background:0,0,0,0,0,0,0,\n0,0,0,0,background player:1,0,0,\n0,0,0,0,0,0,0,\n", 0, "1627766697714.8755"]
	],
	[
		"right [ horizontal TestObject1 | perpendicular TestObject1 ] -> [ perpendicular TestObject1 | ] produces an error #498",
		["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\nright [ right player | right player ] -> [ right player | down player ]\nright [ horizontal Player | perpendicular Player ] -> [ perpendicular Player | ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...pp..\n.......\n\n", [3], "background:0,0,0,0,0,0,0,\n0,0,0,0,background player:1,0,0,\n0,0,0,0,0,0,0,\n", 0, "1627766801845.8809"]
	],
	[
		"[ orthogonal a | moving a ] -> [ moving a | orthogonal a ] produces an error #496",
		["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\n\n======\nRULES\n======\n\nright [ down player | down player ] -> [ down player | action player ]\nright [ orthogonal player | moving player ] -> [ moving player | orthogonal player ]\n\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...pp..\n.......\n\n", [2], "background:0,0,0,0,0,0,0,\n0,0,0,background player:1,0,0,0,\n1,0,0,0,0,0,0,\n", 0, "1627766972788.5994"]
	],
	[
		"1st alternative test for: right [ vertical Player | perpendicular Player ] -> [ perpendicular Player | ] produces error #682",
		["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\ncrate\nblue\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = crate\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer,crate\n\n======\nRULES\n======\n\n[crate]->[up crate]\nright [ horizontal Player | perpendicular crate ] -> [ perpendicular crate | ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...p*..\n.......\n\n", [3, "restart", 3], "background:0,0,0,0,0,0,0,\n0,0,background crate:1,0,0,0,0,\n0,0,0,0,0,0,0,\n", 0, "1627767245523.3818"]
	],
	[
		"2nd alternative test for: right [ vertical Player | perpendicular Player ] -> [ perpendicular Player | ] produces error #682",
		["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\ncrate\nblue\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\n* = crate\ncrateorbackground = crate or Background\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer,crate\n\n======\nRULES\n======\n\n[crate]->[up crate]\nright [ horizontal Player | perpendicular crateorbackground ] -> [ perpendicular crateorbackground | ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...p*..\n.......\n\n", [3], "background:0,0,0,0,0,0,0,\n0,0,background crate:1,0,0,0,0,\n0,0,0,0,0,0,0,\n", 0, "1627767313362.127"]
	],
	[
		"super tricky (related to #469) right [ vertical playerortarget | vertical player ] -> [ vertical playerortarget  |  vertical playerortarget ] ",
		["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\ntarget\nblue\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nplayerortarget = player or target\n@ = target\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\ntarget\n======\nRULES\n======\n\n[target]->[up target]\n\nright [ vertical playerortarget | vertical player ] -> [ vertical playerortarget | vertical playerortarget ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...@p..\n.......\n", [0], "background:0,0,0,0,0,0,0,\n0,0,background target:1,0,0,1,0,\n0,0,0,0,0,0,0,\n", 0, "1627768267232.6797"]
	],
	[
		"right [ vertical playerortarget | vertical player ] -> [ vertical player  |  vertical playerortarget ]",
		["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\nverbose_logging\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN\n\nPlayer\nBlack \n\ntarget\nblue\n\n=======\nLEGEND\n=======\n\n. = Background\nP = Player\nplayerortarget = player or target\n@ = target\n=======\nSOUNDS\n=======\n\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\ntarget\n======\nRULES\n======\n\n[target]->[down target]\n\nright [ vertical playerortarget | vertical player ] -> [ vertical player | vertical playerortarget ]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\n.......\n...@p..\n.......\n", [0], "background:0,0,0,0,0,0,0,\n0,0,background player:1,0,0,0,0,\nbackground target:2,0,0,0,0,0,0,\n", 0, "1627768474619.1826"]
	],
	[
		"gallery game: sokodig",
		[{file:"test_scripts/gallery_game_sokodig.txt"}, [3, 3, 3, 3, 3, 2, 2, 1, 1, 2, 2, 3, 3, 2, 2, 2, 1, 1, 1, 2, 1, 0, 1, 1, 2, 3, 0, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0], {file:"test_endstates/gallery_game_sokodig.txt"}, 24, "1627769218300.0015"]
	],
	[
		"gallery game: at the hedges of time",
		[{file:"test_scripts/gallery_game_at_the_hedges_of_time.txt"}, [1, 1, 0, 0, 0, 0, 3, 3, 2, 3, 3, 3, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 1, 1, 1, 0, 1, 0, 0, 0, "restart", 0, "restart", 0, 3, 3, 3, 0, "undo", 4, 4, 4, 4, 0, 0, 3, 3, 3, 2, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 3, 3, 3, 3], {file:"test_endstates/gallery_game_at_the_hedges_of_time.txt"}, 8, "1627769376598.7021"]
	],
	[
		"gallery game: shall we golf?",
		[{file:"test_scripts/gallery_game_shall_we_golf.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", 2, "tick", 2, "tick", 2, "tick", 2, "tick", "tick", 3, "tick", 3, "tick", 3, "tick", 3, "tick", 3, "tick", "tick", 0, "tick", "tick", 0, "tick", 4, "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", 1, "tick", 1, "tick", 1, "tick", 1, "tick", "tick", "tick", 1, "tick", 0, 4, "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "restart", "tick", "tick", "tick", 3, "tick", 2, "tick", "tick", "tick", 3, 2, "tick", 2, "tick", "tick", 2, 3, "tick", 3, "tick", 3, "tick", 3, "tick", "tick", 3, "tick", 0, "tick", "tick", "tick", 4, "tick", "tick", "tick", 2, "tick", 1, "tick", 1, "tick", 1, "tick", 1, "tick", "tick", 1, 0, "tick", 4, "tick", "tick", "tick", "tick", 2, "tick", "tick", 4, "tick", "tick", "tick", "tick", 3, "tick", 3, "tick", 3, "tick", 3, "tick", 3, "tick", 0, "tick", "tick", 0, "tick", 0, "tick", 0, "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", 4, "tick", "tick", 3, "tick", 2, "tick", "tick", 2, "tick", 2, "tick", 2, 1, "tick", 4, "tick", "tick", "tick", "tick", 3, "tick", 3, "tick", 0, "tick", 0, "tick", 0, "tick", 4, "tick", 4, "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_game_shall_we_golf.txt"}, 2, "1627770891133.2458"]
	],
	[
		"gallery game: Indigestion",
		[{file:"test_scripts/gallery_game_indigestion.txt"}, [0, 2, 2, 3, 0, 0, 0, 1, "restart", 3, 3, 0, 0, 0, 2, 2, 2, 2, 1, 0, 0, 3], {file:"test_endstates/gallery_game_indigestion.txt"}, 13, "1627769708001.6777"]
	],
	[
		"gallery game: two worlds",
		[{file:"test_scripts/gallery_game_two_worlds.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", 2, "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", 2, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 1, "tick", 4, "tick", "tick", "tick", 3, "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_game_two_worlds.txt"}, "7", "1627769807359.2505"]
	],
	[
		"gallery game: mad queens",
		[{file:"test_scripts/gallery_game_mad_queens.txt"}, [0, 2, 3, 3, 0, 1, 3, 0, 0, 0, 1, 0, 0, "undo"], "threatened white:0,black threatened:1,white:2,black:3,2,3,pawn white:4,3,\n1,rook white:5,1,0,1,0,1,0,\n0,1,2,3,2,1,4,3,\n1,0,3,2,3,0,black pawn:6,2,\n0,1,alice white:7,3,2,1,2,3,\n1,0,3,2,3,2,3,2,\n5,1,0,1,0,1,0,1,\n1,0,3,2,3,2,6,2,\n", 4, "1627769866824.53"]
	],
	[
		"gallery game: fall leaves",
		[{file:"test_scripts/gallery_game_fall_leaves.txt"}, [3, 1, 1, 1, 2, 1, 1, 1, 2, 3, 2, 0, 0, 1, 2, 3, "undo", "undo", "undo", 2, 3, 3, 1, 3, 0, 1, 1, 1, 2, 2, 2, 3, 0, 1, 0, 3, 3, 1, 2, 1, 2, 3], {file:"test_endstates/gallery_game_fall_leaves.txt"}, 3, "1627770803266.147"]
	],
	[
		"gallery game: mini nomerads",
		[{file:"test_scripts/gallery_game_mini_nomerads.txt"}, [3, 3, "undo", 1, 2, 3, 2, 1, 3, 4, 4, 0, 1, 0, 3, 3, "undo", "undo", 0, 0, 3, 2, 2, 1, 2, 3, 3, 3, 2, 3, 0, 1, 1, 1, 0, 3], "background wall:0,0,0,0,0,0,0,0,background:1,1,1,\n1,1,0,0,1,1,1,1,1,0,0,\n1,1,1,background lugenbg:2,background lugen:3,0,0,1,1,background player:4,1,\n1,0,0,1,1,1,1,1,0,0,background ballup redball:5,\n1,background lublock:6,1,1,0,0,background redball target:7,background target:8,1,1,1,\n0,0,1,1,1,1,1,0,0,1,1,\n1,1,1,0,0,0,0,0,0,0,0,\n", 9, "1627769983359.2385"]
	],
	[
		"gallery game: Broken Abacus",
		[{file:"test_scripts/gallery_game_broken_abacus.txt"}, [2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 1, 1, 2, 1, 1, 0, 0, 1, 1, 3, 3, 3, 3, 2, 1, 1, 1, 1, 0, 0, 3, 1, 2, 3, 3, 1, 0, 0, 0, 3, 3, 3], {file:"test_endstates/gallery_game_broken_abacus.txt"}, 7, "1627770023925.0144"]
	],
	[
		"gallery game: soliquid",
		[{file:"test_scripts/gallery_game_soliquid.txt"}, [2, 2, 3, 3, 0, 3, 2, 1, 0, 1, 2, 3, 2, 2, 2, 2, 2, 1, 0, 0], "background wall:0,0,0,0,0,0,0,0,0,\n0,background:1,1,1,1,1,1,1,0,\n0,0,0,0,0,0,0,0,0,\n0,1,1,1,background liquid:2,background liquid target:3,2,background crate:4,0,\n0,1,1,1,2,3,2,4,0,\n0,1,1,1,1,background target:5,1,4,0,\n0,0,0,0,0,0,0,0,0,\n0,1,1,1,1,1,1,1,0,\n0,0,0,0,0,0,0,0,0,\n", 3, "1627770093173.774"]
	],
	[
		"gallery game: vexd edit",
		[{file:"test_scripts/gallery_game_vexd_edit.txt"}, [3, 3, 4, 0, 0, 0, 3, 4, 0, 4, 3, 0, 0, 4, 0, 4, 1, 1, 1, 2, 4, 2, 4, 2, 3, 0, 4, 3, 3, 2, 1, 1, 0, 1, 0, 1, 0, 3, 3, 2, 3, 3, 2, 2], {file:"test_endstates/gallery_game_vexd_edit.txt"}, 26, "1627770125763.7583"]
	],
	[
		"gallery game: I'm too far gone",
		[{file:"test_scripts/gallery_game_i_m_too_far_gone.txt"}, [1, 1, 1, 2, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3, 1, 1, 0, 1, 1, 0, 3, 3, 2, 1, 1, 0, 3, 1, 3, 3, 0, 3, 2, 1, 2, 3, 3, 1, 1, 1, 1, 0, 3, 3, 3, 0, 3, 2, 1, 2, 3, 1, 1, 1, 1, 0, 3, 3, 3, 0, 3, 2, 1, 2, 3, 1, 1, 1, 1, 0, 3, 3, 3, 0, 3, "undo", "undo", "undo", 0, 3, 2, 2, 1, 2, 3, 3, 3, 3, 3, 3, 3], {file:"test_endstates/gallery_game_i_m_too_far_gone.txt"}, 1, "1627770159927.0508"]
	],
	[
		"gallery game: maera public works",
		[{file:"test_scripts/gallery_game_maera_public_works.txt"}, [3, 3, 0, 1, 1, 0, 0, 1, 0, 3, 0, 1, 1, 0, 3, 3, 1, 3, 2, 2, 3, 3], {file:"test_endstates/gallery_game_maera_public_works.txt"}, 10, "1627770256142.56"]
	],
	[
		"gallery game: path lines",
		[{file:"test_scripts/gallery_game_path_lines.txt"}, [0, 1, 0, 3, 3, 1, 0, 0, 1, 0, 3, 3, 0, 1, 1, 0, 1, 2, 3, 1, 2, 2, 3, 2, 3, 3, 2, 1, 1], {file:"test_endstates/gallery_game_path_lines.txt"}, 6, "1627770350033.7073"]
	],
	[
		"gallery game: interconnection",
		[{file:"test_scripts/gallery_game_interconnection.txt"}, [0, 3, 3, 3, 3, 2, 1, 1, 1, 0, 1, 1, 2, 3, 3, 0, 3, 2], {file:"test_endstates/gallery_game_interconnection.txt"}, 2, "1627770378083.7458"]
	],
	[
		"gallery game: the art of cloning",
		[{file:"test_scripts/gallery_game_the_art_of_cloning.txt"}, [1, 1, 1, 0, 1, 2, 2, 1, 1, 2, 1, 2, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3], "background wall:0,0,0,0,0,0,0,0,0,\n0,background target:1,1,1,1,1,1,1,0,\n0,1,background:2,2,background crate floor:3,3,2,1,0,\n0,1,2,2,3,3,background player:4,1,0,\n0,1,2,2,3,3,background crate:5,1,0,\n0,1,2,2,2,3,3,1,0,\n0,1,2,2,2,3,3,1,0,\n0,1,1,1,1,1,background crate target:6,1,0,\n0,0,0,0,0,0,0,0,0,\n", 2, "1627770402138.347"]
	],
	[
		"galley game: easy enigma",
		[{file:"test_scripts/galley_game_easy_enigma.txt"}, ["tick", "tick", "tick", 3, 3, "tick", 3, "tick", "tick", 3, "tick", 3, "tick", "tick", 4, "tick", "tick", "tick", "tick", 3, "tick", 2, "tick", 3, "tick", 3, 0, "tick", 1, 2, "tick", "tick", "tick", 0, "tick", "undo", "tick", "undo", "tick", "undo", "tick", 0, "tick", 1, 2, "tick", 3, 2, "tick", "tick", 3, "tick", 3, 0, "tick", 1, "tick", "tick", "tick", 0, "tick", 0, 3, "tick", 3, "tick", 3, 2, "tick", 3, 3, 2, "tick", 1, "tick", 1, "tick", "tick", 1, "tick", 0, "tick", 0, 3, "tick", 2, 1, "tick", 1, "tick", 1, 2, 3, "tick", "tick", "tick", 4, "tick", "tick", "tick", "tick", 4, "tick", "tick", "tick", "tick", "tick", 2, "tick", 2, "tick", 2, "tick", 2, "tick", "tick", "tick", "tick"], {file:"test_endstates/galley_game_easy_enigma.txt"}, 0, "1627770452561.3662"]
	],
	[
		"gallery game: coin collector",
		[{file:"test_scripts/gallery_game_coin_collector.txt"}, [2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 3, 3, 3, 3, 1, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 0, 0, 1], {file:"test_endstates/gallery_game_coin_collector.txt"}, 0, "1627770506202.4248"]
	],
	[
		"gallery game: two worlds",
		[{file:"test_scripts/gallery_game_two_worlds_2.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", 2, "tick", "tick", 2, "tick", 2, "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 4, "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_game_two_worlds_2.txt"}, "7", "1627770584313.9392"]
	],
	[
		"gallery: multi-word dictionary",
		[{file:"test_scripts/gallery_multi_word_dictionary.txt"}, [1, 1, 1, 0, 0, 3, 0, 3, 2, 2, 3, 3, 3, 0, 1, 1, 1, 1, 0, 1, 2, 2, 2, 3, 3], "background wall:0,0,0,0,0,0,0,0,background:1,\n1,1,1,background letterc:2,0,0,1,1,1,\n1,background lettert:3,0,0,1,1,1,background player:4,1,\n0,0,1,1,1,background lettera:5,1,0,0,\n1,1,1,1,1,0,0,1,1,\n1,1,1,0,0,1,1,1,1,\n1,0,0,0,0,0,0,0,0,\n", 0, "1627777496017.3152"]
	],
	[
		"gallery: midas",
		[{file:"test_scripts/gallery_midas.txt"}, [3, 3, 3, 0, 3, 3], "background wall:0,0,0,0,0,0,0,background:1,1,1,0,0,0,1,\n1,1,background goldblock:2,0,0,1,1,1,2,0,0,1,1,1,\n2,0,0,1,1,1,2,0,0,1,1,1,0,0,\n0,1,1,background crown playerground:3,background goldingblock:4,0,0,1,1,1,0,0,0,1,\n1,1,background water:5,0,0,1,1,background target:6,5,0,0,1,1,1,\n5,0,0,1,1,1,0,0,0,0,0,0,0,0,\n", 0, "1627777554778.3638"]
	],
	[
		"gallery: heroes of sokoban",
		[{file:"test_scripts/gallery_heroes_of_sokoban.txt"}, [2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 2, 1, 4, 2, 2, 2, 1, 2, 1, 1, 1, 3, 3, 3, 3, 4, 1, 2], {file:"test_endstates/gallery_heroes_of_sokoban.txt"}, 25, "1627777591387.1045"]
	],
	[
		"gallery: cake monsters",
		[{file:"test_scripts/gallery_cake_monsters.txt"}, [1, 1, 1, 1, 3, 1, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 0, 0, 0, 0, 2, 1, 3, 2, 2, 2, 2, 2, 2, 1], "background wall:0,0,0,0,background blackness:1,1,0,0,0,0,\n0,background purplecake:2,background:3,0,0,0,0,background bluecake:4,2,0,\n0,3,3,3,3,3,3,3,background bluemon:5,0,\n0,0,3,0,0,3,0,3,0,0,\n1,0,3,0,background destroyer:6,3,0,3,0,1,\n1,0,3,0,6,3,0,3,0,1,\n0,0,3,0,0,0,0,3,0,0,\n0,3,3,3,3,3,3,3,4,0,\n0,3,4,0,0,0,0,4,2,0,\n0,0,0,0,1,1,0,0,0,0,\n", 33, "1627777633918.221"]
	],
	[
		"gallery: you're pulleying my leg",
		[{file:"test_scripts/gallery_you_re_pulleying_my_leg.txt"}, [2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 2, 2, 3, 3, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 2, 3, 3, 3, 2, 2, 1, 1, 3, 3, 0, 0, 0, 2, 1, 1, 1, 3, 3, 3, 0, 0, 1, 1, 1, 2, 2, 1, 2, 2, 3, 1], {file:"test_endstates/gallery_you_re_pulleying_my_leg.txt"}, 0, "1627777767719.7332"]
	],
	[
		"gallery: beam islands",
		[{file:"test_scripts/gallery_beam_islands.txt"}, ["tick", "tick", "tick", 3, "tick", 3, "tick", 1, "tick", 0, "tick", 3, "tick", 3, "tick", 2, "tick", 2, "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", 2, "tick", "tick", 3, "tick", 0, "tick", 0, 0, "tick", 0, 0, "tick", 1, "tick", 1, 1, "tick", 1, 1, "tick", 1, 2, "tick", 3, "tick", 3, 3, "tick", 2, "tick", 2, "tick", "tick", "tick", "tick", "tick", 3, "tick", 3, "tick", "tick", 2, 1, "tick", 1, "tick", "tick", "tick", "tick", "tick", 0, "tick", 0, "tick", 0, 0, "tick", "tick", "tick"], {file:"test_endstates/gallery_beam_islands.txt"}, 5, "1627777817083.2576"]

	],
	[
		"gallery: singleton traffic",
		[{file:"test_scripts/gallery_singleton_traffic.txt"}, [3, 2, 2, 1, 1, 3, 3, 3, 3, 0, 1, 2, 2, 2, 2, 2, 2], "background indicatorh indicatorv player:0,background indicatorv player:1,1,1,1,1,1,1,1,0,\nbackground indicatorh player:2,background:3,3,3,3,3,3,3,3,2,\n2,3,3,3,3,3,3,3,3,2,\n2,3,3,3,3,3,3,3,3,2,\n2,3,3,3,3,3,3,3,1,0,\n2,3,3,3,3,3,3,3,3,2,\n2,3,3,3,3,3,1,1,1,0,\n2,3,3,3,3,3,3,3,3,2,\n2,3,3,3,3,3,3,3,3,2,\n0,1,1,1,1,1,1,1,1,0,\n", 6, "1627777857936.9624"]
	],
	[
		"gallery: aa I'm being attacked by a giant tentacle",
		[{file:"test_scripts/gallery_aa_i_m_being_attacked_by_a_giant_tentacle.txt"}, ["tick", 0, 0, "tick", 3, 3, "tick", 3, 3, "tick", 2, 2, 3, 2, "tick", 2, 3, "tick", 3, "tick", 0, "tick"], {file:"test_endstates/gallery_aa_i_m_being_attacked_by_a_giant_tentacle.txt"}, 5, "1627777889846.471"]
	],
	[
		"gallery: censored version of NSFW game pornography for beginners",
		[{file:"test_scripts/gallery_censored_version_of_nsfw_game_pornography_for_beginn.txt"}, [0, 0, 0, 3, 0, 1, 1, 1, 2, 2, 2, 1, 0, 0, 3, 3, 3, 3, 2, 3, 3, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 0, 3, 3, 2, 3, 0, 0, 1, 1, 1, 0, 0, 3, 3], {file:"test_endstates/gallery_censored_version_of_nsfw_game_pornography_for_beginn.txt"}, 3, "1645471800404.333"]
	],
	[
		"gallery: swap bot",
		[{file:"test_scripts/gallery_swap_bot.txt"}, [0, 0, 3, 3, 3, 2, 2, 4, 1, 1, 4, 3, 4, 1, 4, 3, 3, 1, 4, 4, 2, 1, 1, 3, 1, 1, 2, 2, 3, 3, 3, 0, 0, 3, 3, 2, 2, 3, 3, 0, 0, 3, 3, 3, 4, 2, 2, 1, 4, 1, 1, 3, 2, 2, 0, 1, 0], {file:"test_endstates/gallery_swap_bot.txt"}, 12, "1627778061578.234"]
	],
	[
		"gallery: spaceokoban",
		[{file:"test_scripts/gallery_spaceokoban.txt"}, [1, 0, 3, 2], "background wall:0,0,0,0,0,0,0,0,0,0,0,0,\nbackground crate target:1,0,0,0,0,0,0,0,1,0,0,0,\n0,background:2,2,2,background target:3,0,2,0,0,2,2,2,\nbackground player:4,0,2,0,0,0,2,2,2,0,2,0,\n0,2,2,2,2,0,2,0,0,2,2,2,\n2,0,2,0,0,2,2,2,2,0,2,0,\n0,2,2,2,2,0,2,0,0,2,2,2,\nbackground crate:5,0,0,0,0,0,0,0,0,0,0,0,\n", 2, "1627778099494.4666"]
	],
	[
		"gallery: pretender to the crown",
		[{file:"test_scripts/gallery_pretender_to_the_crown.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", 2, "tick", "tick", 2, "tick", 3, "tick", "tick", 3, "tick", 2, "tick", "tick", 2, "tick", "tick", 2, "tick", 2, "tick", "tick", 2, "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", 0, "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_pretender_to_the_crown.txt"}, 13, "1627778178860.0981"]
	],
	[
		"gallery: boxes & balloons",
		[{file:"test_scripts/gallery_boxes_balloons.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", 1, "tick", "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", 1, "tick", 3, "tick", "tick", 3, "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "restart", "tick", "tick", 1, "tick", "tick", 1, "tick", 1, "tick", "tick", 1, "tick", 1, "tick", "tick", 1, "tick", 1, "tick", "tick", 1, "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 0, "tick", "tick", 0, "tick", "tick", "tick", "tick", 0, "tick", 0, "tick", "tick", 0, "tick", 3, "tick", "tick", 3, "tick", "tick", 3, "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_boxes_balloons.txt"}, 16, "1627778214486.7048"]
	],
	[
		"gallery: pushcat junior",
		[{file:"test_scripts/gallery_pushcat_junior.txt"}, ["tick", "tick", "tick", 2, 2, "tick", 2, 2, "tick", 3, "tick", 3, 2, "tick", 3, "tick", 3, 3, "tick", "tick", 0, "tick", 0, "tick", 3, "tick", 0, "tick", 1, 2, "tick", 2, "tick", 2, "tick", 3, 3, "tick", 0, "tick", 3, 3, "tick", 2, "tick", 3, 1, "tick", "tick", 1, "tick", 3, "tick"], {file:"test_endstates/gallery_pushcat_junior.txt"}, 3, "1627778261438.5264"]

	],
	[
		"gallery: sokoboros",
		[{file:"test_scripts/gallery_sokoboros.txt"}, [3, 3, 0, 0, 0, 1, 1, 1, 3, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 2, 3, 3, 3, 0, 1, 2, 2, "undo", "undo", "undo", "undo", 3, 2, 3, 0, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 3, 2, 2, 2], {file:"test_endstates/gallery_sokoboros.txt"}, 1, "1627778292986.5747"]
	],
	[
		"gallery: marble shot",
		[{file:"test_scripts/gallery_marble_shot.txt"}, [0, 3, 3, 3, 2, 3, 3, 0, 3, 0, 0, 0, 0, 1], {file:"test_endstates/gallery_marble_shot.txt"}, 7, "1627778376637.0903"]
	],
	[
		"gallery:cyber-lasso",
		[{file:"test_scripts/gallery_cyber_lasso.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", 4, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", 3, "tick", "tick", "tick", 4, "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 4, "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_cyber_lasso.txt"}, 2, "1627778403808.806"]
	],
	[
		"gallery: train braining",
		[{file:"test_scripts/gallery_train_braining.txt"}, [3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0], {file:"test_endstates/gallery_train_braining.txt"}, 5, "1627778440298.4922"]
	],
	[
		"gallery: enqueue",
		[{file:"test_scripts/gallery_enqueue.txt"}, [3, 0, 4, 1, 1, 4, 3, 4, 1, 4, 2, 4, 3, 4, 4, 4], {file:"test_endstates/gallery_enqueue.txt"}, 4, "1627778464558.232"]
	],
	[
		"gallery: tidy the cafe",
		[{file:"test_scripts/gallery_tidy_the_cafe.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", 1, "tick", 0, "tick", "tick", 3, "tick", "tick", 0, "tick", "tick", 3, "tick", "tick", 3, 0, "tick", 3, "tick", 2, 2, 1, "tick", 1, "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_tidy_the_cafe.txt"}, 2, "1627778539761.4514"]

	],
	[
		"gallery: flower dance",
		[{file:"test_scripts/gallery_flower_dance.txt"}, [3, 3, 0, 1, 2, 2, 3, 3, 0, 1], {file:"test_endstates/gallery_flower_dance.txt"}, 3, "1627778570211.9675"]
	],
	[
		"gallery: stairways",
		[{file:"test_scripts/gallery_stairways.txt"}, [0, 3, 3, 0, 2, 1, 3, 2, 2, 1, 0, 0, 3, 1, 3, 2, 2, 3, 0], "background wall:0,0,0,0,0,0,0,\n0,background:1,background block:2,1,1,1,0,\n0,2,1,2,1,0,0,\n0,1,1,2,2,0,0,\n0,2,background block stairs:3,background player:4,2,0,0,\n0,0,0,2,0,0,0,\n0,0,0,0,0,0,0,\n", 3, "1627778592073.838"]
	],
	[
		"gallery: rock, paper, scissors",
		[{file:"test_scripts/gallery_rock_paper_scissors.txt"}, [0, 0, 0, 0, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 2, 1, 1, 3, 3, 2, 2, 1, 0, 3, 0, 1, 2, 1, 0], {file:"test_endstates/gallery_rock_paper_scissors.txt"}, 8, "1627778623784.798"]

	],
	[
		"gallery: hazard golf",
		[{file:"test_scripts/gallery_hazard_golf.txt"}, [2, 3, 4, 0, 3, 1, 3, 0, 0, 0, 0, 2, 2, 3, 3, 3, 1, 4, 2, 3, 3, 3, 0, 0, 4], {file:"test_endstates/gallery_hazard_golf.txt"}, 6, "1627778649364.9995"]
	],
	[
		"gallery: collapsable sokoban",
		[{file:"test_scripts/gallery_collapsable_sokoban.txt"}, [3, 3, 3, 0, 0, 1, 2, 3, 3, 2, 1, 2, 2, 3, 3, 3, 3], "background:0,0,0,background wall:1,0,0,0,0,0,0,\n0,0,0,0,1,0,0,0,1,0,\n0,0,0,0,1,1,0,1,0,0,\n0,0,0,0,0,0,1,1,1,0,\n0,0,0,0,0,1,1,1,1,0,\n0,0,0,0,1,0,0,0,1,0,\n0,background target:2,0,0,1,1,0,0,0,1,\n0,background crate:3,0,0,0,0,0,0,0,background player:4,\n0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,\n", 3, "1627778695403.4497"]
	],
	[
		"gallery: skipping stones to lonely homes",
		[{file:"test_scripts/gallery_skipping_stones_to_lonely_homes.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 2, "tick", "tick", 2, "tick", 3, "tick", "tick", 3, "tick", "tick", "tick", 2, "tick", 3, "tick", 0, "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", 3, "tick", "tick", 3, "tick", "tick", "tick", "tick", 0, "tick", 0, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_skipping_stones_to_lonely_homes.txt"}, 0, "1627778728427.8804"]
	],
	[
		"gallery: the undertaking",
		[{file:"test_scripts/gallery_the_undertaking.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], "background wall:0,0,0,0,0,0,0,0,0,0,0,\n0,background:1,0,1,0,1,0,1,0,1,0,\n0,1,1,1,0,0,0,1,0,1,0,\n0,1,1,1,1,1,0,0,0,1,0,\n0,0,1,background player:2,1,1,1,1,0,0,0,\n1,1,1,1,1,1,1,1,1,1,background exit:3,\n0,0,1,1,1,1,1,1,0,0,0,\n0,1,1,1,1,1,0,0,0,1,0,\n0,1,1,1,0,0,0,1,0,1,0,\n0,1,0,1,0,1,0,1,0,1,0,\n0,0,0,0,0,0,0,0,0,0,0,\n", 31, "1627778765711.7197"]
	],
	[
		"gallery: JAM3 Game",
		[{file:"test_scripts/gallery_jam3_game.txt"}, [0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 3, 0], {file:"test_endstates/gallery_jam3_game.txt"}, 3, "1627778784846.1284"]
	],
	[
		"gallery: dharma dojo demake",
		[{file:"test_scripts/gallery_dharma_dojo_demake.txt"}, [0, 1, 0, 1, 1, 2, 1, 2, 1, 1, 2, 1, 0, 0, 1, 1], "background ceiling:0,background chute:1,1,1,1,1,1,\n1,1,1,1,background feeder:2,0,background:3,\n3,3,3,3,3,background block2:4,background block1:5,\n4,background block4:6,background hopper:7,0,3,3,3,\n3,3,3,background player:8,6,6,5,\n7,0,3,3,3,3,3,\n3,3,3,3,5,7,0,\n3,3,3,3,3,3,3,\n3,3,4,background wall:9,0,3,3,\n3,3,3,3,3,3,3,\n3,9,0,9,9,9,9,\n9,9,9,9,9,9,9,\n", 3, "1627778809256.5396"]
	],
	[
		"gallery: vines",
		[{file:"test_scripts/gallery_vines.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", 3, "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", 1, "tick", "tick", 0, "tick", "tick", 0, "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_vines.txt"}, 0, "1627778846524.428"]
	],
	[
		"gallery: unconventional guns",
		[{file:"test_scripts/gallery_unconventional_guns.txt"}, [2, 0, 4, 0, 3, 2, 3, 0, 1], {file:"test_endstates/gallery_unconventional_guns.txt"}, 15, "1627778871030.2454"]

	],
	[
		"gallery: newton's crates",
		[{file:"test_scripts/gallery_newton_s_crates.txt"}, [1, 1, 1, 1, 1, 3, 0, 0, 0, 0, 0, 2, 0, 1, 2, 1, 0, 0, 1, 1, 1, 1, 0, 0, 3, 3, 0, 3, 0, 0, 3, 3], {file:"test_endstates/gallery_newton_s_crates.txt"}, 12, "1627778916845.6555"]

	],
	[
		"gallery: count mover",
		[{file:"test_scripts/gallery_count_mover.txt"}, [0, 2, 2, 2, 2, 3, 2, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 3, 2], {file:"test_endstates/gallery_count_mover.txt"}, 1, "1627778943492.7725"]

	],
	[
		"gallery: boxes love boxing gloves",
		[{file:"test_scripts/gallery_boxes_love_boxing_gloves.txt"}, ["tick", "tick", "tick", "tick", 3, "tick", 2, "tick", 2, "tick", 3, "tick", 3, "tick", 0, "tick", "tick", 0, "tick", 3, "tick", 3, "tick", 0, "tick", "tick", "tick", "tick", "tick", 2, "tick", 3, "tick", 3, 0, "tick", "tick", "tick"], {file:"test_endstates/gallery_boxes_love_boxing_gloves.txt"}, 16, "1627778966225.3145"]
	],
	[
		"gallery: vacuum",
		[{file:"test_scripts/gallery_vacuum.txt"}, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 0, 3, 2, 2, 3, 2, 3, 3, 3, 3], {file:"test_endstates/gallery_vacuum.txt"}, 4, "1627779001776.2905"]

	],
	[
		"gallery: paralands",
		[{file:"test_scripts/gallery_paralands.txt"}, [3, 0, 3, 3, 2, 3, 3, 3, 1, 1, 1, 0, 1, 2, 1, 2, 1, 1, 4, 3, 0, 3, 3, 2, 3, 3, 3, 1, 1, 1, 1, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 4], {file:"test_endstates/gallery_paralands.txt"}, 16, "1627779079985.3457"]

	],
	[
		"gallery: Some lines were meant to be crossed",
		[{file:"test_scripts/gallery_some_lines_were_meant_to_be_crossed.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", 2, "tick", "tick", 3, "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", 3, "tick", "tick", 3, "tick", "tick", "tick", "tick", 2, "tick", "tick", 1, "tick", "tick", 0, "tick", "tick", "tick", "tick", 2, "tick", "tick", 2, "tick", 2, "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", 3, "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", 0, "tick", "tick", "tick", "tick", 0, "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", 1, "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_some_lines_were_meant_to_be_crossed.txt"}, 5, "1627779102633.4878"]

	],
	[
		"gallery: Stand Off",
		[{file:"test_scripts/gallery_stand_off.txt"}, [3, 4, 1, 0, 0, 0, 0], {file:"test_endstates/gallery_stand_off.txt"}, 4, "1627779129731.5793"]

	],
	[
		"gallery: 🍡 -ooo- 🍡",
		[{file:"test_scripts/gallery_ooo.txt"}, [2, 2, 2, 1, 3, 0, 0, 0, 0, 1, 1, 0, 0, 3, 2, 2, 2, 2, 3, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3, 2, 2, 3, 3, 0, 0, 0, 0, 3, 3, 3, 0, 1, 1], {file:"test_endstates/gallery_ooo.txt"}, 5, "1627779151924.9846"]

	],
	[
		"gallery: Rose",
		[{file:"test_scripts/gallery_rose.txt"}, [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 3, 3, 4, 1, 4, 3, 1, 4, 3, 1, 1, 1, 1, 4, 1, 1, 4, 3], {file:"test_endstates/gallery_rose.txt"}, 0, "1627779185736.9275"]
	],
	[
		"gallery: season finale",
		[{file:"test_scripts/gallery_season_finale.txt"}, [3, 3, 2, 2, 2, 3, 3, 3, 0, 3, 3, 1, 1, 0, 1, 4, 2, 2, 4, 3, 3, 0, 4, 2, 1, 1, 1, 4, 3, 2, 3, 4, 1, 1, 0, 4, 3, 3, 3, 3, 1, 1, 1, 1, 4, 4, 3, 1, 1, 1, 1, 3, 1, 1, 3, 3, 2, 1, 4, 3, 3, 4, 0, "undo", 1, 1, 1, 3, 3, 0, 0, 3, 1, 3, 4, 3, 3, 3, 3, 3, 2, 2, 2, 1, "undo", 2, "undo", 0, "undo", 2, "undo", "undo", "undo", "undo", 2, 2, 2, 2, 2, "undo", "undo", "undo", 0, 3, "undo", "undo", "undo", "undo", "undo", "undo", 3, 2, 3, 2, 2, 1, 2, "undo", 2, "undo", 0, 3, 3, "undo", "undo"], {file:"test_endstates/gallery_season_finale.txt"}, 98, "1627779217716.8293"]

	],
	[
		"gallery: Spider's Hollow",
		[{file:"test_scripts/gallery_spider_s_hollow.txt"}, [3, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 2], "background backgroundblack:0,0,0,0,0,background wall:1,1,1,0,1,1,\n1,1,1,1,background entrance:2,1,1,1,background block:3,background:4,4,\n4,4,4,4,1,1,1,4,1,4,1,\n1,1,1,0,1,4,1,4,1,background web1:5,4,\n1,0,1,4,1,4,1,5,5,1,0,\n1,4,1,4,4,4,4,1,0,1,background spider web1:6,\n1,1,1,1,1,1,0,1,5,background player web3:7,1,\n4,background hole:8,1,0,0,1,5,4,4,4,1,\n1,0,0,1,1,1,1,1,1,0,0,\n", 10, "1627779381588.6055"]

	],
	[
		"gallery: i herd u liek water templs",
		[{file:"test_scripts/gallery_i_herd_u_liek_water_templs.txt"}, ["tick", "tick", "tick", 1, "tick", "tick", 1, "tick", 1, "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", 3, "tick", "tick", 2, "tick", 1, "tick", "tick", 0, "tick", 1, "tick", 1, "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_i_herd_u_liek_water_templs.txt"}, 7, "1627779397370.8157"]

	],
	[
		"gallery: PrograMaze",
		[{file:"test_scripts/gallery_programaze.txt"}, [1, 1, 3, 3, 2, 4, 3, 4, 3, 4, 2, 4, 3, 4, 0, 3, 3, 4, 3, 0, 1, 4], {file:"test_endstates/gallery_programaze.txt"}, 5, "1627779463747.9348"]

	],
	[
		"gallery: Symbolism",
		[{file:"test_scripts/gallery_symbolism.txt"}, ["tick", 3, 3, 4, 2, 2, "tick", 4, 3, 3, 4, 3, "tick", 0, 4, 3, 4, 2, 4, "tick", 1, 0, 4, 0, 4, 1, 4, "tick", 2, 4, 1, 4, 0, 4, 1, "tick", 1, 4, 1, 2, 2, 2, 1, "tick", 2, 3, 2, 4, "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_symbolism.txt"}, 16, "1627779484111.158"]

	],
	[
		"gallery: The Flames",
		[{file:"test_scripts/gallery_the_flames.txt"}, [3, 0, 0, 3, 3, 0, 3, 3, 2, 2], "background wall:0,0,0,0,0,background:1,0,0,0,0,\n0,0,0,1,1,1,0,0,0,0,\n0,1,1,1,1,1,0,0,0,0,\n1,1,background barrel:2,1,1,0,0,0,1,1,\nbackground fireper1:3,ash background:4,3,1,0,0,background crate:5,1,4,1,\n1,1,1,0,0,1,1,1,background player:6,5,\n1,1,0,0,background pit:7,0,1,1,1,1,\n0,0,0,7,0,0,0,0,0,0,\n0,0,background pit target:8,0,0,0,0,0,0,0,\n", 80, "1627779570245.4102"]

	],
	[
		"gallery: Watch Your Step",
		[{file:"test_scripts/gallery_watch_your_step.txt"}, [3, 3, 2, 1, 1, 0, 0, 3, 3], "background wall:0,0,0,0,0,0,0,0,0,\n0,background:1,1,1,1,1,1,1,0,\n0,1,1,1,1,1,1,1,0,\n0,1,1,1,1,1,1,1,0,\n0,1,background crate0:2,background floor1:3,background floor2:4,background floor4:5,1,1,0,\n0,1,1,background floor0:6,2,2,1,1,0,\n0,1,1,background player:7,2,2,1,1,0,\n0,1,1,1,1,1,1,1,0,\n0,0,0,0,0,0,0,0,0,\n", 3, "1627779589660.2917"]

	],
	[
		"gallery: Spikes 'n' Stuff",
		[{file:"test_scripts/gallery_spikes_n_stuff.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", 2, 3, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "undo", "tick", "tick", "tick", "undo", "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_spikes_n_stuff.txt"}, 0, "1627779607469.4893"]

	],
	[
		"gallery: XL-Plan",
		[{file:"test_scripts/gallery_xl_plan.txt"}, [3, 0, 1, 1, 2, 2, 3, 1, 0, 0, 0, 3, 1, 3, 1, 2, 1, 0, 2], {file:"test_endstates/gallery_xl_plan.txt"}, 6, "1627779632918.6895"]

	],
	[
		"gallery: Play Mini Gemini Replay (PMGRP)",
		[{file:"test_scripts/gallery_play_mini_gemini_replay_pmgrp.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", 1, "tick", "tick", 0, "tick", "tick", 1, "tick", "tick", "tick", 3, "tick", 2, "tick", 1, "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", 0, 4, "tick", 4, "tick", 4, "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/gallery_play_mini_gemini_replay_pmgrp.txt"}, 33, "1627779684110.5146"]

	],
	[
		"vertebrae",
		[{file:"test_scripts/vertebrae.txt"}, [3, 2, 2, 1, 0, 0, 1, 1, 3, 0, 0, 3, 3, 3, 0, 3, 3, 3, 0, 0, 1, 2, 2, 1], {file:"test_endstates/vertebrae.txt"}, 0, "1627779869171.2134"]

	],
	[
		"increpare game: cooperacing",
		[{file:"test_scripts/increpare_game_cooperacing.txt"}, [0, 0, 4, 1, 3, 3, 2, 4, 0, 3, 3, 0, 1, 1, 1, 0, 3, 1, 0, 3, 2, 2, 0, 3, 1, 1, 1, 0, 1, 0, 3, 3, 0, 1, 1, 4, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 2, 1, 1, 1], {file:"test_endstates/increpare_game_cooperacing.txt"}, 5, "1627888788733.95"]

	],
	[
		"increpare game: LED Challenge",
		[{file:"test_scripts/increpare_game_led_challenge.txt"}, [0, 0, "undo", 3, 3, 3, 3, 2, 3, 0, 0, "undo", "undo", 3, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 2, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, "undo", 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 2, 3, 3, 0, 2, 1, 1, 1, "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "restart", 3, 3, 3, 3, 0, 3, 3, 0, 0, 1, 2, 3, 2, 2, 2, 1, 1, 0, 3, 1, 0, 0, 3, 2, 3, 3, 3, 2, 1, 1, 2, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 3, 0, "undo", "undo", 3, 0, 1, 2, 2, 1, 0], "background:0,0,0,0,0,0,0,0,0,0,0,0,\nbackground battery_h light light_left light_right:1,0,0,0,background red_on:2,background cable_ul light light_left light_up:3,background player:4,0,0,0,0,0,\n0,background wall:5,5,0,0,5,5,5,0,0,5,0,\n0,0,0,0,0,0,background green_off:6,0,0,0,0,0,\n0,0,0,background cable_ur:7,0,0,0,0,0,0,0,0,\n", 3, "1627888832451.9211"]

	],
	[
		"increpare game: robot arm",
		[{file:"test_scripts/increpare_game_robot_arm.txt"}, [3, 2, 3, 1, 1, 0, 0, 0, 3, 1, 2, 3, 3, 2, 3, 0, 3, 3, 0, 2, 1, 1, 1, 0, 0, 0, 0, 3, 3, 2, 1, 1, 2, 2, 2, 3, 3, 1, 4], {file:"test_endstates/increpare_game_robot_arm.txt"}, 3, "1627890876560.982"]

	],
	[
		"increpare game: snortal",
		[{file:"test_scripts/increpare_game_snortal.txt"}, [1, 1, 2, 2, 3, 0, 4, 3, 3, "restart", 4, 3, 4, 3, 3, 3, 1, "undo", "undo", 3, 0, 2, 1, 1, 2, 4, 2, 3, 4, 3, 3, 3, 3], "background portalmade:0,background wall:1,1,1,1,1,1,\n1,0,1,background floor:2,background floor snakebod:3,3,3,\nbackground floor red snakeup:4,background portalupr wall:5,background counter:6,1,2,3,2,\n3,2,1,6,1,3,3,\nbackground snakebod:7,3,2,1,6,1,2,\n2,3,3,2,1,6,1,\n2,2,3,3,2,1,6,\n1,1,1,1,background portalleftb wall:8,1,1,\n", 0, "1627888906100.3608"]

	],
	[
		"increpare game: you complete me",
		[{file:"test_scripts/increpare_game_you_complete_me.txt"}, [3, 3, 3, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 0, 0, 1, 1], {file:"test_endstates/increpare_game_you_complete_me.txt"}, 0, "1627888927321.9932"]

	],
	[
		"increpare game: ",
		[{file:"test_scripts/increpare_game.txt"}, [3, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 3, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3], {file:"test_endstates/increpare_game.txt"}, 9, "1627888950120.6284"]

	],
	[
		"increpare game: happy kraken",
		[{file:"test_scripts/increpare_game_happy_kraken.txt"}, [0, 3, 3, 1, 2, 3, 0, 3, 0, 0, 3, 2, 2, 1, 1, 0, 1, 3, 2, 3, 2], {file:"test_endstates/increpare_game_happy_kraken.txt"}, 1, "1627888997152.3042"]

	],
	[
		"increpare game: match three billiards",
		[{file:"test_scripts/increpare_game_match_three_billiards.txt"}, [3, 3, 3], "background wall:0,0,0,0,background wallul:1,background walldl:2,0,0,0,0,\n0,0,0,1,background:3,3,2,0,0,0,\n0,0,1,3,3,3,3,2,0,0,\n0,1,3,3,3,3,3,3,2,0,\n1,3,3,background ball:4,3,3,4,3,3,2,\nbackground wallur:5,3,3,3,3,3,background player:6,3,3,background walldr:7,\n0,5,3,3,3,3,3,3,7,0,\n0,0,5,4,3,3,3,7,0,0,\n0,0,0,5,3,3,7,0,0,0,\n0,0,0,0,5,7,0,0,0,0,\n", 3, "1627889017355.2273"]

	],
	[
		"newton",
		[{file:"test_scripts/newton.txt"}, [0, 3, 3, 3, 0, 1], {file:"test_endstates/newton.txt"}, 5, "1627891250265.7375"]
	],
	[
		"unewton",
		[{file:"test_scripts/unewton.txt"}, [1, 1, 0, 0, 3, 3, 0, 3, 2, 3, 1, 2, 2, 3, 1, 0, 4, "restart", 1, 1, 0, 0, 0, 2, 3, 4, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 4], {file:"test_endstates/unewton.txt"}, 6, "1627891297042.881"]
	],
	[
		"Match two two by twos",
		[{file:"test_scripts/match_two_two_by_twos.txt"}, [3, 0, 1, 4, 3, 4, 3, 4, 2, 4, 1, 4, 1, 4, 1, 4, 0, 4, 0, 4, 3, 4, 3, 4, 3, 4, 3, 2, 1, 4, 1, 4, 1, 4, 1, 1, 4, 0, 4, 3, 1, 1, 3, 4, 3, 4, 3, 4, 2, 1, 1, 1, 2, 4, 3, 4, 3, 4, 3, 1, 1, 0, 0, 3, 3, 2, 2, 4, 1, 1, 1, 2, 4, 3, 3, 4, 0, 0, 3, 3, 4, 1, 1, 1, 1, 2, 4, 3, 3, 1, 1, 3, 3, 3, 2, 4, 1, 1, 4, 0, 3, 3, 3, 4, 1, 1, 4, 3, 4, 1, 1, 4, 3, 3, 4, 0, 4, 2, 4], "background wall:0,0,0,0,0,0,0,0,0,\nbackground midgobble:1,background:2,2,2,2,background gobble red:3,0,0,2,\n0,0,0,0,0,0,0,2,0,\nbackground moved:4,4,4,2,0,0,2,0,4,\n4,4,2,0,0,2,0,4,4,\n4,2,0,0,2,0,4,background blue moved:5,background moved player:6,\n2,0,0,2,0,2,background green:7,background yellow:8,2,\n0,0,2,0,0,0,0,0,0,\n", 4, "1627891318219.0203"]
	],
	[
		"Garten der Medusen",
		[{file:"test_scripts/garten_der_medusen.txt"}, [1, 0, 1, 0, 1, 0, 3, 0, 1, 2, 3, 0, 1, 2, 2], {file:"test_endstates/garten_der_medusen.txt"}, 11, "1627891371908.893"]
	],
	[
		"Kreiseln",
		[{file:"test_scripts/kreiseln.txt"}, [0, 3, 3, 0, 0, 0, 1, 1, 0], "background wall:0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,\nbackground:1,1,1,1,background door_r:2,1,1,\n1,1,1,1,0,0,1,\n1,1,background door_d:3,background hinge:4,1,1,1,\n1,1,1,0,0,1,background target:5,\n1,background player:6,background door_l:7,1,2,1,1,\n1,1,0,0,1,1,1,\n1,1,1,4,background door_u:8,1,1,\n1,0,0,1,1,1,1,\n1,1,7,background crate:9,1,1,1,\n0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,\n", 1, "1627891389385.0288"]
	],
	[
		"Des Poseidons Dreizack",
		[{file:"test_scripts/des_poseidons_dreizack.txt"}, [3, 3, 2, 3, 3, 1, 1, 1, 2, 2, 2, 0, 3, 3, 3, 2, 2, 1, 1, 2, 1, 0, 2, 2, 3, 2, 3, 0, 0, 1, 0, 3, 2, 3, 0, 0, 0, "restart", 2, 3, 2, 3, 3, 2, 2, 2, 1, 2, 1, 2, 1, 1, 0, 3, 3, 2, 3, 0, 0, 1, 0, 3, 2, 3, 0, 0, 2], {file:"test_endstates/des_poseidons_dreizack.txt"}, 3, "1627891415131.3423"]
	],
	[
		"Bring the ice cube to the goal without exposing it to heat.",
		[{file:"test_scripts/bring_the_ice_cube_to_the_goal_without_exposing_it_to_heat.txt"}, [0, 0, 0, 1, 1, 2, 1, 2, 1, 1, 3, 3, 1, 1, 1, 3, 3, 2, 3, 3, 2, 3, 3, 0, 1, 1, 1, 1, 1, 0, 1, 2, 0, 1, 1, 2, 3, 3], {file:"test_endstates/bring_the_ice_cube_to_the_goal_without_exposing_it_to_heat.txt"}, 1, "1627891487429.5186"]
	],
	[
		"der Hydra Krypta",
		[{file:"test_scripts/der_hydra_krypta.txt"}, [1, 1, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 3, 3, 3, 0, 0, 0, 3, 0, 0, 3, 0, 0, "undo", 0, "restart", 0, 0, 0, 0, 3, 0, 0, 0, 0], {file:"test_endstates/der_hydra_krypta.txt"}, 1, "1627891504088.934"]
	],
	[
		"Die schoene Steinmetzin",
		[{file:"test_scripts/die_schoene_steinmetzin.txt"}, [1, 0, 3, 0, 2, 2, 2, 2, 2, 1, 0, 3, 3, 3, 1, 0, 0, 3, 1, 2, 3, 1, 1, 1, 0, 0, 0, 0, 1, 2, 3, 1, 1, 1, 1, 3, 3, 2, 1, 1, 1, 3, 3, 3, 3, 3, 2, 3, 3, 1, 1, 0, 3, 3, 0, 0], {file:"test_endstates/die_schoene_steinmetzin.txt"}, 11, "1627891524914.4753"]
	],
	[
		"I want to grind myself into dust",
		[{file:"test_scripts/i_want_to_grind_myself_into_dust.txt"}, [2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 3, 0, 0, 0, 3, 3, 0, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 1, 1, 1, 1, 0, 0, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 0, 0, 0], {file:"test_endstates/i_want_to_grind_myself_into_dust.txt"}, 3, "1627891570621.5903"]
	],
	[
		"Karamell",
		[{file:"test_scripts/karamell.txt"}, [1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 0, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 3, 3, 3, 3, 3, 2, 2, 2, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 2, 2, 1, 2, 2, 3, 3, 3, 3, 3, 0, 0, 3, 3, 1, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1], {file:"test_endstates/karamell.txt"}, 3, "1627891595796.583"]
	],
	[
		"I am a gust of wind",
		[{file:"test_scripts/i_am_a_gust_of_wind.txt"}, [2, 2, 1, 1, 0, 0, 0, 1, 0, 3, 1, 1, 1, 0, 0, 3, 1, 0, 2, 3, 0, 3, 0, 1, 2, 2, 3, 3, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 2, 2, 3], "background:0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n0,background crate:1,0,0,0,1,0,background target:2,0,\n0,0,0,0,0,background crate target:3,0,0,0,\n0,0,0,2,0,0,0,0,0,\n0,background player:4,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n", 3, "1627891616657.899"]
	],
	[
		"Gabelstapler",
		[{file:"test_scripts/gabelstapler.txt"}, [0, 3, 3, 0, 1, 1, 1, 0, 0, 3, 2, 0, 2, 2, 1, 0, 0, 3, 2, 0, 1, 2, 2, 2, 3, 3, 2, 3, 0, 2, 2, 3, 3, 3, 0, 0, 0, 3, 2, 1, 2, 2, 2, 3, 0, 0, 3, 1], {file:"test_endstates/gabelstapler.txt"}, 5, "1627891651612.1494"]
	],
	[
		"Slot Machine",
		[{file:"test_scripts/slot_machine.txt"}, [3, 3, 2, 3, 0, 1, 2, 2, 3, 3, 3, 0, 1, 1, 2, 1, 0, 1, 1, 0, 0, 3, 0, 3, 2, 2, 1, 2, 3], {file:"test_endstates/slot_machine.txt"}, 3, "1627891674117.2878"]
	],
	[
		"Participating in society",
		[{file:"test_scripts/participating_in_society.txt"}, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 0, 3, 0, 3, 0, 3, 0, 0], {file:"test_endstates/participating_in_society.txt"}, 2, "1627891686718.0212"]
	],
	[
		"Individuals can produce a society, but can a society produce individuals?",
		[{file:"test_scripts/individuals_can_produce_a_society_but_can_a_society_produce.txt"}, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 2, 2, 2, 1, 1, 0, 0, 0, 1, 1, 1, 2, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3], {file:"test_endstates/individuals_can_produce_a_society_but_can_a_society_produce.txt"}, 1, "1627891699152.948"]
	],
	[
		"the exit is under my left foot",
		[{file:"test_scripts/the_exit_is_under_my_left_foot.txt"}, [0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 0, 0, 1, 1, 0, 0, 0, 3, 3, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], {file:"test_endstates/the_exit_is_under_my_left_foot.txt"}, 1, "1627891726028.5537"]
	],
	[
		"I SURE LOOK TASTY!",
		[{file:"test_scripts/i_sure_look_tasty.txt"}, [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3], {file:"test_endstates/i_sure_look_tasty.txt"}, 1, "1627891749225.5503"]
	],
	[
		"Oh no I accidentally swallowed myself!",
		[{file:"test_scripts/oh_no_i_accidentally_swallowed_myself.txt"}, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0, 0, 3, 3, 0, 1, 2, 1, 0, 0, 3, 0, 1, 1], {file:"test_endstates/oh_no_i_accidentally_swallowed_myself.txt"}, 1, "1627891767692.0078"]
	],
	[
		"A cliche for bedtime",
		[{file:"test_scripts/a_cliche_for_bedtime.txt"}, [0, 0, 3, 3, 3, 0, 1, 0, 0, 0, 3, 3, 0, 3, 3, 3, 2, 2, 1, 0, 3, 3, 3, 3], "background:0,0,0,0,0,0,0,0,background target:1,background crate target:2,2,1,2,0,0,background crate:3,3,\n3,3,1,0,3,0,0,3,3,2,0,0,0,0,0,0,0,\n0,0,1,2,2,2,2,0,0,1,0,3,0,1,0,0,background player target:4,\n2,2,2,2,0,0,3,0,0,0,0,0,0,2,1,1,1,\n0,0,0,3,3,3,3,2,0,0,2,1,1,1,0,0,0,\n0,0,0,0,0,0,0,2,2,2,2,2,0,0,2,0,2,\n0,2,0,0,2,0,2,0,2,0,0,0,0,0,0,0,0,\n", 1, "1627891782371.159"]
	],
	[
		"no heroes necessary",
		[{file:"test_scripts/no_heroes_necessary.txt"}, [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 0, 0], {file:"test_endstates/no_heroes_necessary.txt"}, 7, "1627891794816.7026"]
	],
	[
		"Sokogoban",
		[{file:"test_scripts/sokogoban.txt"}, [0, 0, 2, 1, 3, 0, 1, 0, 1, 1, 1, 0, 3, 0, 3, 3, 2, 1, 2, 2, 0, 0, 3, 3, 2, 2, 3, 3, 2, 2, 2, 1, 1, 1, 0], {file:"test_endstates/sokogoban.txt"}, 4, "1627891805914.7593"]
	],
	[
		"SHY SNAKE",
		[{file:"test_scripts/shy_snake.txt"}, [3, 2, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 0, 0, 1, 0, 1], {file:"test_endstates/shy_snake.txt"}, 3, "1627891845531.74"]
	],
	[
		"a distant sunset",
		[{file:"test_scripts/a_distant_sunset.txt"}, [3, 0, 0, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3, 3, 3, 0, 1, 1, 4], {file:"test_endstates/a_distant_sunset.txt"}, 1, "1627891893861.785"]
	],
	[
		"Seize the flag!",
		[{file:"test_scripts/seize_the_flag.txt"}, [3, 0, 4, 0, 4, 1, 3, 2, 1, 1, 4, 2, 3, 1, 1, 0, 0, 3, 1], {file:"test_endstates/seize_the_flag.txt"}, 1, "1627891910482.2163"]
	],
	[
		"An OK multiban level",
		[{file:"test_scripts/an_ok_multiban_level.txt"}, [1, 1, 0, 0, 3, 0, 3, 3, 2, 2], "background wall:0,0,0,0,0,0,0,background:1,1,\n1,1,0,0,1,background crate:2,0,1,0,\n0,1,2,0,1,0,0,background target:3,3,\n3,3,0,0,1,1,0,1,0,\n0,1,2,0,1,0,0,1,1,\nbackground player:4,2,0,0,0,0,0,0,0,\n", 1, "1627891934752.9468"]
	],
	[
		"Unclean Residues",
		[{file:"test_scripts/unclean_residues.txt"}, [2, 3, 2, 0, 1, 1, 3, 2, 2, 1, 0, 2, 2, 1, 0, 1, 0, 3], "background wall:0,0,0,0,0,0,0,0,\n0,0,background:1,1,1,1,1,1,\n1,0,0,1,1,1,1,1,\n1,1,0,0,1,1,0,background c_d c_l c_r c_u crate:2,\nbackground player target:3,1,1,0,0,1,1,1,\nbackground c_d c_l crate target:4,background c_u crate target:5,1,1,0,0,1,1,\n1,1,1,1,1,0,0,1,\n1,1,1,1,1,1,0,0,\n0,0,0,0,0,0,0,0,\n", 4, "1627891984006.569"]
	],
	[
		"Car Crash",
		[{file:"test_scripts/car_crash.txt"}, [2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, "restart", 1, 1, 1, 2, 2, 3, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1, 1, "restart", 3, 3, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 3, 3, 2, 2, 2, 3, 2, 3, 3, 3, 3, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1, 1], {file:"test_endstates/car_crash.txt"}, 1, "1627891999778.8188"]
	],
	[
		"Alternatey",
		[{file:"test_scripts/alternatey.txt"}, [0, 1, 2], "background player:0,0,0,0,0,0,\n0,background goal player:1,0,0,1,0,\n0,0,background:2,2,0,0,\n0,0,0,0,0,0,\n0,1,0,0,1,0,\n0,0,2,2,0,0,\n", 4, "1627892045163.187"]
	],
	[
		"Push Pull",
		[{file:"test_scripts/push_pull.txt"}, [2, 1, 0, 3, 3, 3, 2, 1, 0, 1, 1, 2, 2, 3, 0, 1, 0, 3, 1, 2, 0, 0, 3, 0], "background wall:0,0,0,0,0,0,\n0,0,background:1,background crate2:2,background target:3,1,\n1,0,0,3,background player:4,background crate:5,\n1,1,0,0,0,0,\n1,1,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 1, "1627892056284.9724"]
	],
	[
		"Slide Pull",
		[{file:"test_scripts/slide_pull.txt"}, [1, 0, 1, 1, 2, 0, 1, 1, 1, 3, 3, 3, 2, 1, 2, 3, 0, 3, 0, 3, 3, 3, 1, 1, 2, 1, 0, 0, 3, 3, 3, 2, 1], "background wall:0,0,0,0,0,background:1,0,1,\nbackground crate:2,1,0,1,0,1,background crate target:3,1,\n0,1,0,1,background target:4,1,0,1,\n0,1,4,2,0,0,0,1,\nbackground player:5,1,1,0,0,1,1,1,\n1,0,0,0,0,0,0,0,\n", 5, "1627892069950.7256"]
	],
	[
		"Make way!",
		[{file:"test_scripts/make_way.txt"}, [0, 3, 0, 0, 0, 1, 1, 0, 1, 2], "background wall:0,0,0,0,0,0,0,\n0,0,0,0,background:1,background pit:2,background player:3,\n1,1,1,1,background crate target:4,0,0,\n1,2,2,1,1,1,1,\n1,0,0,1,background crate:5,2,5,\n1,2,2,background target:6,0,0,1,\n1,2,2,2,2,1,1,\n0,0,1,1,1,1,1,\n1,1,6,0,0,0,0,\n0,0,0,0,0,0,0,\n", 1, "1627892084709.0234"]
	],
	[
		"DICE GOBLIN",
		[{file:"test_scripts/dice_goblin.txt"}, [3, 0, 3, 3, 3, 3, 3], {file:"test_endstates/dice_goblin.txt"}, 3, "1627892093944.9868"]
	],
	[
		"KONSUM",
		[{file:"test_scripts/konsum.txt"}, [3, 1, 4, 3, 3, 2, 3, 4, 2, 4, 1, 4, 2, 2, 1, 1, 4, 4, 3, 3, 0, 0, 0, 0, 0, 3, 3, 3, 4, 4, 0, 4, 2, 2, 2, 1, 1, 3, 4, 3, 4, 4, 0, 0, 0, 4, 2, 2, 1, 1, 1, 4, 0, 1, 1, 4, 0, 4, 2, 1, 4, 3, 4, 2, 2, 2, 2, 4, 2, 4, 3, 0, 0, 1, 4, 2, 2, 4, 0, 0, 0, 0, 3, 1, 2, 2, 2, 2, 0, 2, 4, 3, 0, 0, 3, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 0, 3, 4, 3, 3, 0, 0, 0, 0, 2, 4, 3, 4, 0, 1, 2, 4, 3, 0, 4, 1, 1, 0, 4, 4, 1, 0, 4], {file:"test_endstates/konsum.txt"}, 1, "1627892104859.3337"]
	],
	[
		"Expand, avoid the flames [also it's not always solvable]",
		[{file:"test_scripts/expand_avoid_the_flames_also_it_s_not_always_solvable.txt"}, [1, 0, 0, 1, 1, 1, 0, 4, 3, 3, 0, 4, 1, 4, 0, 4, "undo", 4, "undo", 4, "undo", 4], {file:"test_endstates/expand_avoid_the_flames_also_it_s_not_always_solvable.txt"}, 1, "1627894530268.055"]
	],
	[
		"Pupush",
		[{file:"test_scripts/pupush.txt"}, [1, 0, 0, 3, 3, 1, 1, 1, 2, "undo", 3, 2, 1, 3, 3, 3, 0, 1, 1, 1, 1, 1, 2, 2, 3, 1, 0, 3, 3, 3, 3, 1, 1, 1, 1, 3, 0, 0, 3, 3, 3, "restart", 2, 1, 1, 3, 3, 3, 0, 1, 1, 0, 1, 1, 1, 1, 2, 3, 3, 2, 3, 0, 0, 0, 3, 0, 1, 2], "background wall1 wall2:0,0,0,0,0,0,0,0,0,\n0,background vwall1 vwall2:1,background:2,2,2,1,2,2,2,\n2,0,1,2,background ziel:3,2,1,2,2,\nbackground target:4,2,0,1,2,2,2,1,2,\nbackground crate1:5,2,2,0,1,2,background crate2:6,2,2,\nbackground tuer1 tuer2 tuer_offen:7,2,background player1:8,2,0,1,2,2,2,\n1,2,background player2:9,2,2,0,1,2,3,\n2,1,2,2,4,2,0,1,2,\n2,2,1,2,2,2,2,0,0,\n0,0,0,0,0,0,0,0,0,\n", 1, "1627892193346.9036"]
	],
	[
		"GAPFILLER / LUECKEFUELLER",
		[{file:"test_scripts/gapfiller_lueckefueller.txt"}, [1, 0, 0, 3, 3, 2, 2, 2, 2, 2, 2, 3, 1, 1, 1, 0, 2, 2, 2, 0, 0, 2], "background wall:0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,\nbackground:1,0,0,0,background landed poo:2,0,0,\n0,0,2,2,2,1,0,\nbackground landed player:3,2,2,2,0,0,2,\n0,2,0,0,0,background landed:4,0,\n1,0,0,2,2,2,2,\n2,2,4,1,1,0,0,\n0,0,1,0,0,0,4,\n0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,\n", 11, "1627892223719.676"]
	],
	[
		"pushit",
		[{file:"test_scripts/pushit.txt"}, [1, 0, 3, 0, 3, 3, 3, 2, 1, 1, 2, 3, 3, 1, 0, 3, 2, 1, 2, 1, 2, 1, 1, 1, 1, 0, 0, 0], "background wall:0,0,0,0,0,0,0,0,background:1,\n1,1,1,1,0,0,background crate:2,background player:3,background poo:4,\n4,1,0,0,1,background poo target:5,1,background target:6,1,\n0,0,1,2,4,2,1,0,0,\n1,5,4,background crate target:7,1,0,0,4,1,\n1,1,1,0,0,1,1,4,1,\n1,0,0,0,0,0,0,0,0,\n", 5, "1627892234377.7625"]
	],
	[
		"AUTOHURDLER",
		[{file:"test_scripts/autohurdler.txt"}, [0, 3, 1, 2, 2, 3, 3, 0, 3, 0, "undo", "undo", "undo", 1, 1, 0, 3, "undo", 3, 2, 2, 2, 1, 1, 1, 2, 3, 2, 2, 1, 1, 1, 1, 0, 3, 1, 2], {file:"test_endstates/autohurdler.txt"}, 1, "1627892250172.2253"]
	],
	[
		"Snakey Snakey Makes A Sixy",
		[{file:"test_scripts/snakey_snakey_makes_a_sixy.txt"}, [0, 0, 3, 2, 2, 4, 2, 1, 1, 0, 0, 3, 3, 2, 3, 0, 0, 0, 1, 1, 2, 3, 0, 0, 1, 2, 2, 2, 1, 1, 0, 3, 0, 1, 2, 2, 2, 1, 0, 3, 2, 1, 0, 3, 3, 2, 1, 1, "undo", "undo", 1, 1, 0, 3, 3, 0, 1, 2, 1, 1, 0, 3, 2, 1, 0, 3, 3, 2, 2, 1, 0, 1, 0, 3, 2, 3, 3, 3, 0, 1, 2, 3, 3, 0, 1, 2, 1, 0, 3, 2, 1, 1, 0, 3], "1 background:0,0,0,3 background:1,0,0,0,0,0,\n0,0,0,2 background:2,0,0,0,0,0,\n0,0,3 background c_down c_right prev1:3,1 background c_up prev2:4,1,0,0,0,0,\n0,0,1 background c_left player:5,2,0,0,0,0,0,\n0,2,0,2,2,0,0,0,0,\n0,2,2,1,0,2,0,0,0,\n0,0,0,2,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n", 0, "1627892275012.8281"]
	],
	[
		"Voitex Rasteriser [Demake of Vertex Dispenser Puzzle Mode]",
		[{file:"test_scripts/voitex_rasteriser_demake_of_vertex_dispenser_puzzle_mode.txt"}, [0, 4, 3, 2, 4, 1, 4, 4, 1, 4, 2, 4, 3, 4, 2, 4, 1, 1, 4, 0, 4, 0, 3, 0, 4, 1, 4, 2, 4, 1, 4, 2, 4, 2, 4], "background:0,0,0,0,0,0,0,0,0,\n0,0,0,0,1 background:1,2 background:2,3 background player:3,0,0,\n0,0,0,1,2,3 background:4,1,0,0,\n0,0,0,2,1,2,0,0,0,\n0,0,0,1,2,1,2,0,0,\n0,0,0,0,1,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n", 0, "1627892333608.1028"]
	],
	[
		"Voitex Rasteriser 2 [2nd Demake of Vertex Dispenser Puzzle Mode]",
		[{file:"test_scripts/voitex_rasteriser_2_2nd_demake_of_vertex_dispenser_puzzle_mo.txt"}, [0, 4, 3, 3, 4, 2, 2, 1, 4, 0, 4, 0, 0, 0, 4, 2, 4, 1, 4, 1, 4, 0, 4, 3, 4, 0, 4, 3, 4, 4, 3, 2, 4, 4, 3, 4, 2, 4, 4, 2, 4, 4], {file:"test_endstates/voitex_rasteriser_2_2nd_demake_of_vertex_dispenser_puzzle_mo.txt"}, 1, "1627892374590.2808"]
	],
	[
		"All Green and Blue on Yellow",
		[{file:"test_scripts/all_green_and_blue_on_yellow.txt"}, [2, 3, 0, 1, 0, 0, 3, 3, 2, 2, 1, 2, 1, 0, 3, 0, 1, 2, 1, 0, 3, 0, 3, 0, 2, 2, 1, 0], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background cratev:2,2,\n1,0,0,background crateh:3,background player:4,1,\nbackground target:5,1,0,0,1,5,\n5,background crateh target:6,1,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", 3, "1627892394944.1912"]
	],
	[
		"ALL GREEN TO BLUE",
		[{file:"test_scripts/all_green_to_blue.txt"}, [0, 3, 0, 1, 2, 3, 3, 0, 3, 2], "background:0,0,0,0,\nbackground crateh:1,0,background cratev:2,0,\n0,0,0,1,\n1,2,0,1,\n1,1,1,1,\n1,1,0,0,\n0,0,0,0,\n0,0,0,0,\n0,0,0,0,\n1,background player:3,0,0,\n0,0,0,0,\n", 7, "1627892412039.6033"]
	],
	[
		"5-junctions-maker / 5kreuzungenmacher",
		[{file:"test_scripts/5_junctions_maker_5kreuzungenmacher.txt"}, [2, 1, 1, 1, 1, 1, 1, 3, 3, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 3, 3, 1, 3, 0, 1, 2, 2, 2, 1, 0, 0, 3, 3, 2, 2, 2], {file:"test_endstates/5_junctions_maker_5kreuzungenmacher.txt"}, 1, "1627892456324.3245"]
	],
	[
		"NIGHTMARECROBAN",
		[{file:"test_scripts/nightmarecroban.txt"}, [0, 1, 1, 0, 0, 3, 2, 3, 2, 0, 3, 2, "undo", 1, 2, 2, 2, 3, 0, 0, 1, 0, 1, 2, 3, 2, 2, 1, 0], "background:0,background wall:1,1,1,1,1,0,1,background p_d:2,\n2,background p_l:3,1,1,1,background p_r:4,1,3,1,\n1,4,4,background p_l player:5,background p_u:6,1,1,2,background crate p_l:7,\n2,4,1,1,1,background p_u ziel:8,1,1,1,\n0,1,6,background crate p_u:9,4,1,0,1,background p_r ziel:10,\n2,4,1,0,1,1,1,1,1,\n", 7, "1627892520679.5837"]
	],
	[
		"Scalesoban",
		[{file:"test_scripts/scalesoban.txt"}, [3, 0, 0, 1, 1, 1, 0, 1, 2, 2, 3, 2, 1, 0, 0, 1, 1, 2, 2, 3, 3], "background:0,background wall:1,1,1,1,1,0,1,0,\n0,0,1,1,1,0,1,background target:2,1,\n1,0,0,0,background player:3,1,1,0,0,\n0,background crate target:4,1,1,1,background crate:5,1,1,1,\n0,1,0,0,0,1,0,1,0,\n0,0,1,0,1,1,1,1,1,\n", 5, "1627892538124.7505"]
	],
	[
		"Vexatious Match 3",
		[{file:"test_scripts/vexatious_match_3.txt"}, [2, 1, 0, 3, 2, 0, 0, 3, 3, 4, 1, 2, 1, 0, 4, 3, 3, 4, 1, 4, 2, 4, 2, 4, 1, 0, 3, 0, 4, "undo", 2, 1, 4, 1, 2, 3, 4], "2 background:0,background:1,1,1,\n0,5 background:2,5 background selectcuror_top:3,background selectcuror_bottom:4,\n1,1,1,1,\n1,1,1,2,\n", 8, "1627892551428.3667"]
	],
	[
		"SLIME VAT FILLER",
		[{file:"test_scripts/slime_vat_filler.txt"}, [2, 2, 1, 2, 3, 4, 1, 2, 2, 3, 1, 3, 1, 0, 1, 2, 2, 3, 0, 2, 3, 0, 1, 3, 3, 3, 3, 0, 0, 3, 2, 1, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 2, 1, 1, 1, "restart", 2, 2, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 2, 0, 0, 0, 0, 1, 0, 0, 3, 3, 3, 3, 2, 2, 2, 2, 1, 2, 2, 0, 0, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0], "background wall:0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,background:1,1,1,1,\n1,1,1,1,1,1,1,0,0,\n1,1,1,1,1,1,1,1,1,\n1,1,0,0,1,1,1,background blob_u target:2,background blob_v:3,\n3,3,background blob_d:4,1,1,1,0,0,1,\n1,background blob target:5,background player:6,2,3,3,3,4,1,\n1,0,0,1,1,1,5,1,1,\n1,1,1,1,1,0,0,1,1,\n1,1,1,1,1,1,1,1,1,\n0,0,1,1,1,1,1,1,1,\n1,1,1,1,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,\n", 16, "1627892574193.6548"]
	],
	[
		"A CLEAR VIEW OF THE SKY",
		[{file:"test_scripts/a_clear_view_of_the_sky.txt"}, [2, 2, 2, 2, 4], {file:"test_endstates/a_clear_view_of_the_sky.txt"}, 16, "1627892617303.8054"]
	],
	[
		"A CLEAR VIEW OF THE SKY",
		[{file:"test_scripts/a_clear_view_of_the_sky.txt"}, [2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 2, 0, 4, 2, 2, 4, 4, 0, "restart", 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4], {file:"test_endstates/a_clear_view_of_the_sky_2.txt"}, 26, "1627892636599.9646"]
	],
	[
		"BIAXIAL INVASION OF SATURN",
		[{file:"test_scripts/biaxial_invasion_of_saturn.txt"}, [2, 4, 2, 2, 2, 2, 3, 4], {file:"test_endstates/biaxial_invasion_of_saturn.txt"}, 13, "1627892668524.9329"]
	],
	[
		"sum three horizontally to 8!",
		[{file:"test_scripts/sum_three_horizontally_to_8.txt"}, [3, 3, 2, 4], "background wall:0,0,0,0,0,0,\n0,background:1,1,background k3:2,2,0,\n0,1,1,2,background k2:3,0,\n0,1,background player:4,1,1,0,\n0,1,background player_r:5,3,2,0,\n0,0,0,0,0,0,\n", 19, "1627893119768.2104"]
	],
	[
		"Schleimban",
		[{file:"test_scripts/schleimban.txt"}, [3, 0, 0, 1, 1, 1, 0, 1, 2, 3, 2, 3], {file:"test_endstates/schleimban.txt"}, 3, "1627893180878.8218"]
	],
	[
		"Legend of Swixero",
		[{file:"test_scripts/legend_of_swixero.txt"}, [1, 0, 2, 3, 3, 1, 1, 0, 3, 0, 3, 0, 3, 0, 1, 2, 3, 2, 2, 2, 1, 1, 0, 0, 3], "background wall:0,0,0,0,0,0,0,\n0,0,background:1,1,1,1,1,\n1,0,0,1,background crate:2,background target:3,background crate target:4,\n1,1,0,0,1,4,4,\n1,1,1,0,0,1,1,\n1,background player:5,4,1,0,0,1,\n1,1,4,1,1,0,0,\n0,0,0,0,0,0,0,\n", 5, "1627893213826.6091"]
	],
	[
		"Xorro The Chaos Warden",
		[{file:"test_scripts/xorro_the_chaos_warden.txt"}, [1, 2, 1, 0, 1, 2, 3, 3, 1, 0, 0, 3], "background:0,0,0,background wall:1,1,1,1,1,\n1,1,1,0,0,1,1,0,\nbackground crate:2,0,0,0,1,1,0,background target:3,\n0,3,0,1,1,0,background player:4,0,\n0,0,1,1,0,background crate target:5,2,5,\n0,1,1,0,0,0,0,0,\n1,1,1,1,1,1,1,1,\n", 3, "1627893230650.995"]
	],
	[
		"Put the logs in the water, elephant!",
		[{file:"test_scripts/put_the_logs_in_the_water_elephant.txt"}, [2, 3, 0, 0, 0, 3, 2, 1, 2, 0, 3, 3, 2, 4, 1, 0, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 2, 0, 0, 0, 1, 0, 0, 3, 3, 4, 0, 4, 0, 1, 1, 2, 3, 2, 3], {file:"test_endstates/put_the_logs_in_the_water_elephant.txt"}, 19, "1627893532893.2812"]
	],
	[
		"Pocket Gopher: Root-Shoot Nibbler",
		[{file:"test_scripts/pocket_gopher_root_shoot_nibbler.txt"}, [0, 4, 2, 2, 1, 0, 0, 0, 1, 1], {file:"test_endstates/pocket_gopher_root_shoot_nibbler.txt"}, 11, "1627893552932.8503"]
	],
	[
		"North Wind Simple Sailboat Buoy Collection",
		[{file:"test_scripts/north_wind_simple_sailboat_buoy_collection.txt"}, ["tick", 2, 0, "tick", 1, "tick", "tick", 2, "tick", 0, "tick", 3, 3, "tick", 2, "tick", "tick", "tick", "tick", 1, "tick", 0, "tick", "tick", "tick", 3, "tick"], {file:"test_endstates/north_wind_simple_sailboat_buoy_collection.txt"}, 4, "1627893570488.5464"]
	],
	[
		"It gets its Feet Wet",
		[{file:"test_scripts/it_gets_its_feet_wet.txt"}, [3, 0, 0, 3, 1, 2, 2, 2, 2, 0, 3, 3, 4, 0, 0, 0, 3, 4, 1, 1, 0, 3, 1, 4, 2, 2, 4, 3, 2, 2, 4, 0, 0, 4, 1], {file:"test_endstates/it_gets_its_feet_wet.txt"}, 1, "1627893596068.238"]
	],
	[
		"Neoprenanzieher",
		[{file:"test_scripts/neoprenanzieher.txt"}, [3, 3, 3, 2, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 3, 3, 3, 2, 2, 2, 2, 1, 2, 1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1, 1, 1, 1, 3, 2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 2, 2, 2, 2, 1, 1, 2, 2, 1, 0, 2, 2, 0, 0, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 3, 2, 2, 2, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3], {file:"test_endstates/neoprenanzieher.txt"}, 2, "1627893621526.3992"]
	],
	[
		"Moved by leaves of grass",
		[{file:"test_scripts/moved_by_leaves_of_grass.txt"}, [0, 1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3], {file:"test_endstates/moved_by_leaves_of_grass.txt"}, 1, "1627893692010.7708"]
	],
	[
		"Place the crate on the targets",
		[{file:"test_scripts/place_the_crate_on_the_targets.txt"}, [1, 1, 0, 0, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1], {file:"test_endstates/place_the_crate_on_the_targets.txt"}, 1, "1627893701797.368"]
	],
	[
		"Two-Step Pete",
		[{file:"test_scripts/two_step_pete.txt"}, [2, 2, 2, 3, 0, 0, 0, 1, 1, 2, 2], "background wall:0,0,0,0,0,\n0,0,background target:1,background crate target:2,background lastdown player target:3,\nbackground:4,0,0,background crate:5,2,\n5,1,0,0,1,\n2,5,1,0,0,\n0,0,0,0,0,\n", 13, "1627893714599.114"]
	],
	[
		"Microban x 10",
		[{file:"test_scripts/microban_x_10.txt"}, [3, 1, 2, 3, 2, 3, 0, 3, 0, "undo", 0, 3, 0], {file:"test_endstates/microban_x_10.txt"}, 2, "1627893725475.1794"]
	],
	[
		"The Far Away Danish Pastry is always Deliciouser.",
		[{file:"test_scripts/the_far_away_danish_pastry_is_always_deliciouser.txt"}, [3, 2, 1, 0, 3], "background:0,0,background fruit:1,0,0,0,0,1,1,1,0,1,0,\n0,1,0,0,1,0,0,0,0,1,0,0,0,\n0,0,1,0,0,0,0,0,1,1,0,0,0,\n1,0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,background player:2,0,0,0,0,0,0,0,\n0,0,0,0,1,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,1,1,0,0,\n", 13, "1627893739801.0737"]
	],
	[
		"CŌDEX·LŪBRICUS",
		[{file:"test_scripts/c_dex_l_bricus.txt"}, [2, 3, 0, 3, 2, 3, 0, 0, 1, 1, 3, 2, 1, 0, 0, 3, 0, 1, 0, 3, 2], {file:"test_endstates/c_dex_l_bricus.txt"}, 40, "1627893759466.7617"]
	],
	[
		"Oh No My Dog Is About To Swallow A Piece Of Chocolate",
		[{file:"test_scripts/oh_no_my_dog_is_about_to_swallow_a_piece_of_chocolate.txt"}, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], {file:"test_endstates/oh_no_my_dog_is_about_to_swallow_a_piece_of_chocolate.txt"}, 1, "1627893792808.8965"]
	],
	[
		"paint everything everywhere!",
		[{file:"test_scripts/paint_everything_everywhere.txt"}, [1, 0, 3, 0, 1, 2, 3, 2, 1], {file:"test_endstates/paint_everything_everywhere.txt"}, 33, "1627893815569.176"]
	],
	[
		"Putting Bicycle Helmets on Young Children",
		[{file:"test_scripts/putting_bicycle_helmets_on_young_children.txt"}, [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3, 0, 3, 0, 0, 0, 0, 0, 1, 0, 3, 3, 3, 2, 3, 1, 0, 0, 3, 3, 3], {file:"test_endstates/putting_bicycle_helmets_on_young_children.txt"}, 1, "1627893829074.0923"]
	],
	[
		"No! Don't eat that!",
		[{file:"test_scripts/no_don_t_eat_that.txt"}, [2, 0, 1, 3, 0, 2, 1, "restart", 0, 2, 1, 2, 0, 1, 3, 2, 1], {file:"test_endstates/no_don_t_eat_that.txt"}, 7, "1627893851172.3552"]
	],
	[
		"Recondite Star Sector Sigma",
		[{file:"test_scripts/recondite_star_sector_sigma.txt"}, [0, 0, 1, 1, 1, 1, 0, 3, 2, 2, 2, 1, 1, 1, 0, 3, 2, 3, 3, 0, 2, 1, 1, 0, 3, 3, 2, 3, 0], "hintergrund wand:0,0,0,0,0,0,0,0,hintergrund:1,\n1,1,0,0,0,0,1,1,0,\n0,1,hintergrund kiste:2,1,1,0,0,1,0,\nhintergrund spieler:3,hintergrund ziel:4,0,0,0,1,0,0,0,\n0,0,1,0,1,0,0,0,1,\n1,1,0,0,0,0,0,0,0,\n", 15, "1627893876246.9329"]
	],
	[
		"Crate Swarp",
		[{file:"test_scripts/crate_swarp.txt"}, [1, 1, 2, 1, 0, 3, 3, 3, 0, 1, 2, 1, 0, 2, 1, 1, 0, 3, 3], "background wall:0,0,0,0,0,0,0,\n0,0,background:1,1,0,0,0,\n0,1,1,1,1,1,0,\n0,background crate2:2,background player:3,background target2:4,1,1,0,\n0,0,background crate1 target1:5,1,0,0,0,\n0,0,1,1,0,0,0,\n0,0,0,0,0,0,0,\n", 11, "1627893891182.2898"]
	],
	[
		"NO FORBIDDEN SYMBOLS",
		[{file:"test_scripts/no_forbidden_symbols.txt"}, [0, 0, 2, 1, 1, 1, 0, 3, 3], "background:0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,\n0,0,background crate:1,background player:2,0,0,0,\n0,0,background crate horizont:3,1,0,0,0,\n0,0,1,0,0,0,0,\n0,0,0,1,0,0,0,\n0,0,0,0,0,0,0,\n", 1, "1627893988626.8167"]
	],
	[
		"SWIMMING TIME!",
		[{file:"test_scripts/swimming_time.txt"}, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 0, 0, 1], {file:"test_endstates/swimming_time.txt"}, 2, "1627893997860.5815"]
	],
	[
		"(NSFW / censored) Tugging a trapped XXX XXX out from under one's XXX",
		[{file:"test_scripts/nsfw_censored_tugging_a_trapped_xxx_xxx_out_from_under_one_s.txt"}, [1, 3, 2, 1, 0, 2, 2, 3, 0, 3, 1, 0, 0, 0, 1, 0, 3, 3, 1, 1, 3, 2, 2, 1, 3, 3, 2, 1, 3, 1, 3, 2, 3, 2, 0, 3, 3, 2, 1, 0, 1, 3, 1, 1], {file:"test_endstates/nsfw_censored_tugging_a_trapped_xxx_xxx_out_from_under_one_s.txt"}, 7, "1627894024137.8354"]
	],
	[
		"REALTIME DOG MOUNTAIN RESCUE",
		[{file:"test_scripts/realtime_dog_mountain_rescue.txt"}, [0, 0, 3, 2, 2, 2, 3, 3, 2, 2, 2, 3, 2, 2, 0, 0, 1, 3, 4, 4, 2, 2, 2, 4, 4, 3, 3, 2, 2, 2, "tick", 3, 2, 2, 4, 4, 2, 2, 3, 3, 4, 2, 3, 3, 2, 2, 4, 2, 3, 3, 3, 2, 4, 2, 3, 2, 2, 2, 2, 2, 2, 4, 2, 2, 4, 2, 2, 2, 4, 4, 4, 4, 4, "tick", 4, 4, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 2, 2, 2, 3, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 0, 0, 0, 2, 2, 2, 1, 0, 0, 0, 1, 1, 0, 0, 2, 2, "tick", 3, 3, 1, 2, 1, 1, 3, 3, "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, "tick", 2, 2, 2, 2, 1, 2, 2, 2, 2, 3, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0], {file:"test_endstates/realtime_dog_mountain_rescue.txt"}, 7, "1627894053043.232"]
	],
	[
		"You can't make this up!",
		[{file:"test_scripts/you_can_t_make_this_up.txt"}, [2, 1, 1, 1, 1, 0, 0, 0, 1, 2, 2, 1, 2, 2, 2, 3, 3], {file:"test_endstates/you_can_t_make_this_up.txt"}, 1, "1627894109796.508"]
	],
	[
		"Explodoban",
		[{file:"test_scripts/explodoban.txt"}, [1, 1, 2, 1, 1, 0, 4, 2, 4, 1, 2, 1, 4], "background wall:0,0,0,0,0,0,0,0,background:1,\n1,1,0,0,0,1,0,1,0,\n0,1,background crate:2,background target:3,background player:4,0,0,1,1,\n1,1,0,0,0,1,1,0,0,\n0,0,1,3,1,0,0,0,2,\n1,1,0,0,0,0,0,0,0,\n", 11, "1627894122279.9321"]
	],
	[
		"Bicycle-Kick Football",
		[{file:"test_scripts/bicycle_kick_football.txt"}, [3, 0, 0, 4, 2, 2, 1, 3, 0, 4, 1, 4, 2, 4, 3, 0, 4, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 4, 2, 3, 3, 0], {file:"test_endstates/bicycle_kick_football.txt"}, 7, "1627894148704.56"]
	],
	[
		"Van-to-Mobile-Living-Space-Conversion Window-Frame-Installation",
		[{file:"test_scripts/van_to_mobile_living_space_conversion_window_frame_installat.txt"}, [3, 3, 0, 3, 2, 0, 3, 3, 1, 2, 2, 3, 2, 3, 0], {file:"test_endstates/van_to_mobile_living_space_conversion_window_frame_installat.txt"}, 13, "1627894165909.7253"]
	],
	[
		"The sponge what lights up the seafloor",
		[{file:"test_scripts/the_sponge_what_lights_up_the_seafloor.txt"}, [2, 2, 2, 3, 2, 1, 1, 1, 0, 0, 1, 1, 1, 2, 2, 3, 2, 1, 1, 1, 3, 1, 2, 2, 3, 3, 2, 3, 0, 1, 0, 1, 1, 0, 0, 0, 3, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 3, 0, 1, 2, 2, 1, 2, 1, 2, 3, 0, 3, 3, 3, 0, 3, 3, 2, 2, 2, 2, 1, 1, 0, 0, 3, 0, 0, 1, 1], {file:"test_endstates/the_sponge_what_lights_up_the_seafloor.txt"}, 17, "1627894176777.0325"]
	],
	[
		"(censored version of NSFW game) Sexual Intercourse",
		[{file:"test_scripts/censored_version_of_nsfw_game_sexual_intercourse.txt"}, [0, 0, 3, 2, 3, 0, 0, 0, 2, 2, 2, 3, 0, 3, 3, 0, 0, 0, 3, 0, 1, 1, 0, 0, 0, 1, 2, 2, 3, 1, 1, 1, 2, 3, 0, 0, 1, 1, 2, 3, 0, 3, 2, 2, 0, 3, 3, 3, 1, 1, 2, 1, 1, 0, 1, 1, 2, 2, 1, 1, 0, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 3, 2, 2, 1, 1, 0, 1, 1, 2, 1, 0, 0, 2, 0, 3, 2, 3, 3, 2, 2, 2, 3, 0, 3, 3, 0, 3, 0, 0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 2, 1, 3, 0, 0, 0, 3, 1, 2, 2, 3, 3, 3, 3, 0, 2, 3, 3, 3, 3, 2, 3, 3, 3, 0, 1, 0, 0, 3, 0, 0, 1, 0, 1, 1, 1, 2, 1], {file:"test_endstates/censored_version_of_nsfw_game_sexual_intercourse.txt"}, 1, "1645642688116.959"]
	],
	[
		"Attractor Net",
		[{file:"test_scripts/attractor_net.txt"}, [3, 2, 3, 1, 1, 1, 0, 0, 3, 3, 2, 2, 2, 3, 3, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 1, 2, 3, 3, 0, 0, 2, 2, 0, 1, 1, 0, 3, 3, 3, 2, 2, 0, 1, 2, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 3, 3, 3, 3, 3, 2, 2, 1, 0, 1, 2, 1, 0], {file:"test_endstates/attractor_net.txt"}, 13, "1627894355351.2"]
	],
	[
		"Yellow Box",
		[{file:"test_scripts/yellow_box.txt"}, [3, 0, 3, 3, 3, 2, 2], {file:"test_endstates/yellow_box.txt"}, 27, "1627894381565.9717"]
	],
	[
		"Sokoban[Sokoban[Sokoban]]]",
		[{file:"test_scripts/sokoban_sokoban_sokoban.txt"}, [1, 1, 0, 3, 0, 3, 3, 2, 3, 2, 1, 1, 1, 0, 3, 2, 3, 0, 3, 0, 1], {file:"test_endstates/sokoban_sokoban_sokoban.txt"}, 1, "1627894393467.7576"]
	],
	[
		"Varifocal Nightmare",
		[{file:"test_scripts/varifocal_nightmare.txt"}, [2, 1, 0, 0, 0, 1, 1, 2, 3, 3], "background wall:0,0,0,0,0,0,0,0,background target1a:1,background:2,0,\n0,0,0,0,2,2,background crate1b:3,0,0,0,0,\n2,background player:4,2,2,0,0,0,0,background crate1a:5,2,background target1b:6,\n2,0,0,0,0,0,0,0,0,0,background crate2a:7,\n2,0,0,0,0,0,2,2,background target2b:8,0,0,\n0,0,2,2,2,2,0,0,0,0,background player target2a:9,\n2,background crate2b:10,2,0,0,0,0,0,0,0,0,\n", 5, "1627894408429.9158"]
	],
	[
		"Gem Socketeer",
		[{file:"test_scripts/gem_socketeer.txt"}, [0, 2, 3, 0, 3, 2, 2, 3, 3, 2, 1, 1, 1, 2, 3, 1, 1, 3, 2, 2, 1, 2, 3, 0, 1, 1, 0, 2, 3, 0, 0, 3, 0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 2, 2, 3, 2, "undo", 1, 2, 3, 3, 2, 3, 0], {file:"test_endstates/gem_socketeer.txt"}, 5, "1627894805419.4346"]
	],
	[
		"Empty loop test",
		[{file:"test_scripts/empty_loop_test.txt"}, [3, 2, 2, 1, 2, 1, 1, 0, 3], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,1,background player:2,\n1,0,0,1,0,background crate:3,\nbackground crate target:4,1,0,0,1,1,\n4,background target:5,1,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", "1", "1628787178284.0164"]
	],
	[
		"Loop example with both late and regular rules.",
		[{file:"test_scripts/loop_example_with_both_late_and_regular_rules.txt"}, [3], "background crate2:0,background:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\n", 0, "1628792234693.525"]
	],
	[
		"\"right [ Player ] -> [ up Player ]\" gets compiled to down #755",
		["title Simple Block Pushing Game\nauthor David Skinner\nhomepage www.puzzlescript.net\n\ndebug\n\n========\nOBJECTS\n========\n\nBackground\nLIGHTGREEN GREEN\n11111\n01111\n11101\n11111\n10111\n\n\nflag\nDarkBlue\n\n\nWall\nBROWN DARKBROWN\n00010\n11111\n01000\n11111\n00010\n\nPlayer\nBlack Orange White Blue\n.000.\n.111.\n22222\n.333.\n.3.3.\n\nCrate\nOrange\n00000\n0...0\n0...0\n0...0\n00000\n\n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\nO = flag\n\n\n=======\nSOUNDS\n=======\n\nCrate MOVE 36772507\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nflag\nPlayer, Wall, Crate\n\n======\nRULES\n======\n\n\nright [ crate ] [ stationary flag ] -> [ ] [ action flag]\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\no*\n*.\n\n", [4], "background flag:0,background crate:1,\nbackground:2,2,\n", 0, "1629150419859.7258"]
	],
	[
		"in sound section can't mention an object called horizontali - cf #757",
		[{file:"test_scripts/in_sound_section_can_t_mention_an_object_called_horizontali.txt"}, [0, 1, 2, 2], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background target:2,background player:3,\nbackground crate:4,0,0,2,1,1,\n1,1,0,0,0,0,\n1,4,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1629314889294.314"]
	],

	[
		"VEXT EDIT",
		[{file:"test_scripts/gallery_game_vexd_edit.txt"}, [0, 3, 0, 3, 2, 1, 1, 2, 2, 3, 0, 1, 0, 3, 3, 1, 2, 2, 2, 2, 3, 0, 0, 1, 0, 0, 3, 3, 0, 2, 2, 0, 3, 1, 0, 2, 2, 2, 1, 1, 1, 2], {file:"test_endstates/vext_edit.txt"}, 6, "1642084928380.6565"]
	],
	[
		"VEXT EDIT B",
		[{file:"test_scripts/gallery_game_vexd_edit.txt"}, [0, 3, 3, 3, 2, 2, 4, 1, 0, 3, 4, 0, 0, 0, 0, 4, 3, 3, 2, 2, 2, 2, 1, 4, 0, 0, 0, 3, 0, 4, 2, 2, 2, 2, 3, 3, 4, 1, 4, 0, 0, 0, 0, 0, 4, 1, 2, 2, 2, 2, 2, 4, 0, 0, 0, 3, 0, 4, 4, 4, 1, 4, 2, 1, 0, 3, 3, 4, 2, 1, 0, 4, 1, 0, 0, 3, 3, 3, 2, 1, 1, 1, 3, 4, 2, 2, 1, 1, 1, 0, 3, 4, 3, 3, 2, 2, 2, 1, 2, 1, 4, 0, 0, 0, 3, 3, 0, 0, 0, 4, 3, 2, 2, 1, 0, 4, 1, 2, 1, 4, 3, 3, 0, 4, 2, 3, 0, 4, 2], {file:"test_endstates/vext_edit_b.txt"}, 49, "1642083700844.0618"]
	],
	[
		"Undo and Real-time #796",
		[{file:"test_scripts/undo_and_real_time_796.txt"}, ["tick", 4, 4, 4, 1, 1, "tick", 1, "tick", "tick", "tick", "tick", "undo", "undo", "tick", "tick", "tick", "undo"], {file:"test_endstates/undo_and_real_time_796.txt"}, 1, "1642738088805.1099"]
	],
	[
		"Cucumber Surprise",
		[{file:"test_scripts/cucumber_surprise.txt"}, [1, 3, 3, 0, 1, 0, 2, 2, 3, 0, 1, 0, 0, 0, 2, 3, 0, 1, 2, 2, 2, 1, 1, 0, 3, 3, 2, 3, 0, 1, 0, 0, 0, 0], "background wand:0,0,0,0,0,0,0,\n0,0,0,background:1,1,1,1,\n1,0,0,background ziel:2,2,1,0,\n1,1,0,0,1,0,1,\n0,1,1,0,0,background player:3,1,\n1,1,1,1,0,0,1,\nbackground katze:4,background katze push_up:5,background gurke:6,1,1,0,0,\n0,0,0,0,0,0,0,\n", 35, "1645544413590.2168"]
	],
	[
		"Chevron Lodger",
		[{file:"test_scripts/chevron_lodger.txt"}, [3, 2, 2, 1, 1, 1, 0, 3, 3, 2, 3, 3, 0, 0, 0, 3, 0, 3, 0, 3, 3, 2, 2, 2, 0, 1, 1, 3, 2, 2, 2, 1, 1, 0, 0, 1, 1, 2, 1, 1, 3, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 2, 1, 2, 1, 1, 1, "undo", "undo", 3, 0, 1, 1, 2, 1, 1], "background wall:0,0,0,0,0,0,0,0,0,0,0,\n0,0,background:1,1,1,1,0,0,0,0,0,\nbackground crate_top:2,background crate_bottom:3,1,1,0,0,0,0,0,background player_top:4,background player_middle:5,\n5,background player_bottom:6,0,0,0,0,0,1,1,1,1,\n0,0,0,1,1,1,1,1,1,0,0,\n0,1,1,1,1,1,1,0,0,1,1,\n1,1,0,1,1,0,0,1,1,1,1,\n1,1,1,0,0,1,1,1,1,1,1,\n1,0,0,0,0,0,0,0,0,0,0,\n", 9, "1645544442907.0366"]
	],
	[
		"Two-Tone Tango",
		[{file:"test_scripts/two_tone_tango.txt"}, [0, 2, 2, 1, 3, 0, 0, 0, 3, "restart", 0, 1, "restart", 2, 0, 1, 1, 1, 0, 0, 0, 3, 3, 2, 2, 1, 3, 2, 1, 0, 0, 0, 3, 3, 2, 1, 3, 2, 1, 2, 2, 1], "background wall:0,0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,1,\n0,0,0,1,1,background target:3,background crate:4,\nbackground player:5,5,0,0,1,1,1,\n1,1,1,0,0,1,1,\n1,1,0,1,0,0,1,\n1,1,background freund:6,6,1,0,0,\n0,0,0,0,0,0,0,\n", 3, "1645544569499.8286"]
	],
	[
		"Hole-Stuffer",
		[{file:"test_scripts/hole_stuffer.txt"}, [3, 2, 2, 1, 1, 1, 0, 0, 1, 1, 2, 1, 1, 0, 3, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1, 0, 0, 1, 1, 3, 3, 0, 1, 1, 2, 2, 1, 3, 2, 1, 1], "background randommoved wall:0,background wall:1,1,1,1,1,1,1,1,\n1,background:2,2,2,2,background crate:3,2,2,1,\n1,2,2,2,background player player_1:4,background player player_4:5,5,background player player_5:6,1,\n1,2,2,2,background player player_2:7,2,2,3,1,\n1,2,2,2,background player player_3:8,5,5,6,1,\n1,2,2,2,2,2,2,2,1,\n1,2,2,2,2,3,2,2,1,\n1,2,2,2,2,2,2,2,1,\n1,1,1,1,1,1,1,1,1,\n", 3, "1645544598726.4905"]
	],
	[
		"Wand Spinner",
		[{file:"test_scripts/wand_spinner.txt"}, [2, 3, 3, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 0, 0, 3, 3, 0, 3, 3, 2, 2, "undo", 2], "background tl wall:0,background wall:1,1,1,1,1,1,\nbackground:2,2,2,background target_h:3,1,1,1,\n2,background stick_w:4,3,1,1,background target_v:5,5,\nbackground stick_e:6,2,1,1,2,background stick_n:7,background stick_s target_h:8,\n4,1,1,2,2,background player target_h:9,6,\n1,1,1,1,1,1,1,\n", 27, "1645544636691.3945"]
	],
	[
		"Acorn Scorchery Paradise",
		[{file:"test_scripts/acorn_scorchery_paradise.txt"}, [3, 0, 0, 2, 1, 0, 2, 1, 0, 2, 3, 0, 0, 2, 0, 2, 1, 0, 0, 3, 3, 2, 3, 0, 0, "restart", 0, 0, 0, 0, 2, 1, 0, 0, 3, 0, 3, 0, 2, 1, 2, 2, 1, 0, "restart", 1, 0, 2, 3, 0, 1, 2, 3, 3, 0, 2, 1, 1, 0, 0, 2, 3, "undo", "undo", "undo", 3, 0, 0, "restart", 1, 0, 0, 2, 3, 3, 0, 2, 1, 1, 0, 0, 2, 2, 3, 3, 0, 1, 0, 0, 3, 2, 1, 2, 2], "background wall:0,0,0,0,0,\n0,0,background crate target:1,background target:2,background crate:3,\nbackground:4,0,0,4,4,\n4,background player:5,0,0,4,\n4,background crate_hot target:6,4,0,0,\n0,0,0,0,0,\n", 3, "1645544663060.2773"]
	],
	[
		"Match-Maker",
		[{file:"test_scripts/match_maker.txt"}, [2, 2, 3, 3, 0, 1, 0, 3, 1, 2, 2, 2, 2, 2, 3, 0, 0, 0], "background wall:0,0,0,0,0,0,\n0,0,0,background:1,1,1,\nbackground crate:2,1,1,0,0,1,\n1,1,1,1,1,0,\n0,background crate hands_down:3,background crate hands_up:4,background player:5,1,1,\n1,0,0,2,3,4,\n0,0,1,0,0,0,\n0,0,0,0,0,0,\n", 11, "1645544739914.9763"]
	],
	[
		"Pushy-V Pully-H",
		[{file:"test_scripts/pushy_v_pully_h.txt"}, [0, 0, 2, 2, 1, 1, 0, 3, 2, 3, 3, 3, 3, 1, 0, 3, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 3, 3, 0, 3, 3, 2, 1, 1, 0, 1, 1, 0, 0, 0, 3, 1, 0, 0, 3, 3, 3, 2, 1, 2, 2, 2, 3, 2, 3, 0, 0, 0, 0, 1, 1, 2, 1, 1, 0, 0, 1, 3, 2, 3, 0, 1, 1, 1, 2, 2, 2, 2, 2, 0, 2, 3, 3, 3, 0, 1, 3, 3, 2, 1, 1, 0, 0, 0, 2, 0, 0], "background wall:0,0,0,0,0,0,0,\n0,0,background:1,1,1,1,1,\n1,0,0,1,background target:2,background crate:3,background crate target:4,\n1,1,0,0,3,background player:5,2,\n1,2,1,0,0,1,2,\n3,4,3,1,0,0,1,\n1,1,1,1,1,0,0,\n0,0,0,0,0,0,0,\n", 19, "1645544759965.9036"]
	],
	[
		"Crates move when you move",
		[{file:"test_scripts/crates_move_when_you_move.txt"}, [2, 1, 0, 1, 1, 2, "undo", 1, 2, 3, 3, 3, 0, 2], "background wall:0,0,0,0,0,background:1,0,1,\n1,1,0,1,0,1,background target:2,1,\n0,1,0,1,2,1,0,1,\n0,1,2,background crate:3,0,0,0,1,\n3,background player:4,1,0,0,1,1,1,\n3,0,0,0,0,0,0,0,\n", 11, "1645544790426.2693"]
	],
	[
		"Resin-Caster",
		[{file:"test_scripts/resin_caster.txt"}, [3, 0, 0, 1, 2, 0, 1, 1, 1, 2, 2, 3, 0, 1, 0, 3, 3, 1, 2, 2, 1, 1, 1, 0, 0, 3, 3, 3, 3, 3], "background:0,background wall:1,1,1,1,1,0,1,0,\n0,0,1,1,1,0,1,0,1,\n1,background crate2:2,0,0,0,1,1,0,0,\n0,0,1,1,1,background player:3,1,1,1,\n0,1,1,0,background crate1:4,1,0,1,0,\n0,0,1,0,1,1,1,1,1,\n", 18, "1645544805711.4153"]
	],
	[
		"pipe puffer",
		[{file:"test_scripts/pipe_puffer.txt"}, [0, 0, 2, 1, 0, 4, 3, 3, 0, 0, 0, 1, 1, 1, 3, 4, 0, 1, 1, 2, 4, 2, 4, 3, 2, 3, 2, 2, 1, 1, 0, 4, 4, 2, 3, 3, 3, 0, 3, 0, 0, 4, 2, 1, 1], {file:"test_endstates/pipe_puffer.txt"}, 4, "1645544832464.3308"]
	],
	[
		"crate guardian",
		[{file:"test_scripts/crate_guardian.txt"}, [2, 2, 3, 3, 2, 2, 1, 2, 2, 1, 1, 0, 2, 3, 3, 0, 1, 1, 2, 1, 3, 0, 1, 3, 0, 0, 0, 1, 2, 2, 3, 2, 1, 2, 1, 1, 0, 0, 2, 1, 1, 1, 3, 0, 0, 0, 1, 0, 0, 0, 0, 3, 3, 2], "background wall:0,0,0,0,0,0,0,0,0,\n0,background:1,1,1,1,1,1,1,0,\n0,1,1,aura2 background target:2,aura2 background crate:3,2,1,1,0,\n0,1,background player target:4,3,aura2 background statue2:5,3,background target:6,1,0,\n0,background crate:7,1,aura2 background:8,3,3,7,1,0,\n0,1,6,1,1,1,6,1,0,\n0,1,1,6,1,6,1,1,0,\n0,1,1,7,1,1,1,1,0,\n0,0,0,0,0,0,0,0,0,\n", 11, "1645544889350.864"]
	],
	[
		"Don't let your goals slip away",
		[{file:"test_scripts/don_t_let_your_goals_slip_away.txt"}, [0, 0, "undo", "undo", 1, 0, 0, 3, 3, 3, 2, 1, 1, 0, 0, 1, 0, 0, "restart", 0, 0, 0, 2, 3, 3, 2, 1, 1, 2, 1, 0, 0, 0, 0, 0, 2, 0, 1, 1, 1, 3, 3, 3, "undo", "undo", 3, 3, 1, 1, 1, 2, 3, 0, 1, 1, "restart", 0, 0, 1, "restart", 3, 0, 0, 1, 1, 1, 1, "undo", 0, 1, 2, 2, 1, 2, 2, 1, 2, 2, 3, 3, 0, 2, 1, 1, 0, 0, 3, 3, 3, 3, 3, 1, 1], "background:0,background wall:1,1,1,1,1,0,1,0,\n0,0,1,1,1,0,1,0,1,\n1,0,0,0,0,1,1,0,background player:2,\n0,background crate:3,1,1,1,0,1,1,1,\n0,1,background target:4,0,0,1,0,1,background crate target:5,\n0,0,1,0,1,1,1,1,1,\n", 3, "1645544970753.6125"]
	],
	[
		"Eyeball-watching flowers bloom",
		[{file:"test_scripts/eyeball_watching_flowers_bloom.txt"}, [1, 2, 3, 3, 0, "undo", 0, 0, 3, 2, 3, 2, 1, 1, 0, 0, 1, 0, 1, 2, 3], "background:0,background wall_activated:1,1,1,background wall_inactivated:2,0,\n1,0,0,0,background crate:3,1,\n2,3,background player_right strahlen_right:4,0,0,2,\n2,0,background strahlen:5,0,0,2,\n2,0,5,0,0,1,\n0,2,background strahlen_left wall_burnt:6,1,1,0,\n", 3, "1645545001785.496"]
	],
	[
		"Subway upholstry snot smearing championship",
		[{file:"test_scripts/subway_upholstry_snot_smearing_championship.txt"}, [1, 2, 3, 0, 1, 3, 3, 3, 3, 3, 2, 0, 1, 2, 3, 0, 1, 1, 1, 2, 1, 0, 3, 3, 1, 2, 2, 1, 1, 1, 3, 0, 0, 1, 3, 0, 0, 1, 2, 1, 1, 3, 2, 2, 3, 0, 2], "background void:0,background wallt:1,background wall:2,background wallalt:3,2,background wallb:4,0,background walltb:5,background tilled:6,\nbackground crate tilled:7,6,background walltbalt:8,1,4,7,5,6,5,\n8,6,6,7,7,8,5,6,6,\n7,background player:9,5,1,4,6,1,2,4,\n0,8,6,6,7,8,0,5,background:10,\n10,10,5,0,1,2,3,2,4,\n", 8, "1645545059102.073"]
	],
	[
		"Double-Entry Bookkeeping Simulator#1",
		[{file:"test_scripts/double_entry_bookkeeping_simulator_1.txt"}, [1, 1, 1, 1, 3, 3, 0, 0, 3, 0, 3, 3, 3, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 2, 1, 1, 3], "background:0,0,0,0,0,0,background wall:1,1,1,1,1,\n1,0,0,0,0,0,0,1,0,0,0,\n0,1,0,0,0,0,0,0,1,0,background wall wall3:2,\n2,2,1,0,0,0,0,0,0,1,0,\n2,2,2,1,0,0,1,1,1,1,1,\n1,1,0,1,1,1,1,1,background blue1:3,0,0,\n0,0,0,background player:4,0,1,1,0,0,0,0,\n1,0,1,0,0,0,1,1,0,0,0,\n0,0,0,1,0,0,0,1,1,0,background wall wall1:5,\n5,0,0,0,1,1,1,1,1,1,0,\n0,0,0,0,0,1,0,0,0,0,1,\n1,1,1,1,1,1,1,0,0,0,0,\n", 1, "1645545097375.3352"]
	],
	[
		"Double-Entry Bookkeeping Simulator#2",
		[{file:"test_scripts/double_entry_bookkeeping_simulator_1.txt"}, [2, 0, 3, 2, 2, 1], {file:"test_endstates/double_entry_bookkeeping_simulator_2.txt"}, 17, "1645545159763.6345"]
	],
	[
		"short adventure in sticky wall land",
		[{file:"test_scripts/short_adventure_in_sticky_wall_land.txt"}, [2, 3, 0, 3, 0, 0, 2, 1, 0, 2, 1, 2, 2, 3, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 2, 1, 3, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 3, 0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, "undo", "undo", "undo", 0, 1, 0, 3, 2, "undo", 2, 3, 1, 1, 2], {file:"test_endstates/short_adventure_in_sticky_wall_land.txt"}, 1, "1645545192906.1704"]
	],
	[
		"[testing for recording through level-changes A]  Level-Change test",
		[{file:"test_scripts/testing_for_recording_through_level_changes_a_level_change_t.txt"}, [2, 1, 0, 3, 3, 3, 2, 1, 0, 1, 2, 2, "undo", 1, 2, 2, 3, 0, 1, 0, 3, 0, 0, 1, 2, 3, 2, 2, 3, 3, 3, 0, 1, 2, 1, 0, 0, 3, 2, 2, 1, 2, 3, 0, 0, 0, 1, 2], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,1,1,\n1,0,0,1,0,background crate:2,\nbackground crate target:3,1,0,0,1,1,\nbackground player target:4,3,1,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", 0, "1645546672241.5308"]
	],
	[
		"[testing for recording through level-changes B] Simple Block asdfsadf Game",
		[{file:"test_scripts/testing_for_recording_through_level_changes_b_simple_block_a.txt"}, [2, 1, 0, 3, 3, 2, 1, 3, 3, 2, 1, 0, 1, 1, 2, 2, 3, 0, 1, 0, 3, 0, 0, 1, 2, 3, 2, 2, 3, 3, 0, 1, 2, 1, 0, 0], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,1,1,\n1,0,0,1,0,background crate:2,\nbackground target:3,1,0,0,1,background player:4,\nbackground crate target:5,5,1,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", 0, "1645546774769.8335"]
	],

	[
		"[testing for recording through level-changes C] Don't asdlet your goals slip away",
		[{file:"test_scripts/testing_for_recording_through_level_changes_c_don_t_asdlet_y.txt"}, [0, 0, 0, 2, 3, 3, 2, 1, 1, 1, 2, 1, 0, 0, 0], "background:0,background wall:1,1,1,1,1,0,1,0,\n0,0,1,1,1,0,1,0,1,\n1,0,0,0,background crate:2,1,1,0,0,\n0,0,1,1,1,2,1,1,1,\n0,1,background target:3,0,background player:4,1,0,1,0,\n3,0,1,0,1,1,1,1,1,\n", 1, "1645546878695.2927"]
	],

	[
		"Wallnut chasing mouse chasing cat chasing dog chasing man",
		[{file:"test_scripts/wallnut_chasing_mouse_chasing_cat_chasing_dog_chasing_man.txt"}, [1, 0, 0, 0, 3, 0, 1, 2, 2, 2, 3, 0, 3, 3, 2, 1, 0, 1, 2, 3, 0, 1, 1, 1, 1, 1, 0, 3, 2, 3, 3, 1, 0, 2, 0, 3, 2, 2, 1, 0, 3, 0, 3, 2, 3, 2, 3], "background:0,0,background wall:1,1,1,1,0,0,0,0,1,\n0,0,1,1,0,0,0,1,0,0,0,\n1,0,0,1,1,1,0,1,1,1,0,\n1,0,0,0,0,0,1,0,1,0,background mouseactiver:2,\n0,background crate:3,3,1,0,1,0,background catactiver:4,background dogactiver target:5,0,0,\n1,1,1,1,1,background player target:6,0,1,1,1,background wallnutidle:7,\n0,0,0,0,1,0,1,1,1,0,1,\n1,1,0,0,0,1,0,1,0,0,0,\n", 23, "1645578571689.1494"]
	],
	[
		"Kettle, except something is off",
		[{file:"test_scripts/kettle_except_something_is_off.txt"}, [2, 0, 2, 0, 0, 1, 1, 1, 3, 3, 2, 1, 1, 0, 0, 1, 2, 1, 1, 0, 3, 3, 3, 3, 0, 1, 0, 0], "background wall:0,0,0,0,0,0,0,0,0,0,\n0,background:1,1,background rpolice:2,1,1,1,1,1,0,\n0,1,1,background crate:3,1,1,2,1,1,0,\n0,background dpolice:4,3,background upolice:5,1,background target:6,3,1,1,0,\n0,1,1,6,1,1,background lpolice:7,1,1,0,\n0,1,1,1,1,1,6,1,1,0,\n0,4,3,5,6,1,1,1,1,0,\n0,1,1,7,1,1,1,1,1,0,\n0,1,1,1,1,1,1,1,1,0,\n0,0,0,0,0,0,0,0,0,0,\n", 7, "1645562847734.9592"]
	],
	[
		"Lightdown",
		[{file:"test_scripts/lightdown.txt"}, [0, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 1, 2, 1, 0, 3, 0, 0, 1, 1, 2, 3, 2, 3, 3, 0, 0, 3, 1, 3, 2, "undo", 1, 2, 3, "undo", 0, 1, 1, 2, 3, 0, 3, 3, 3, 2, "undo", 0, 3, 1, 2, 2, 3, 3, 3, "undo", "undo", 1, 0, 0, 2, 3, 2, 2, 3, 3, 3, 3, 0, 0, 1, 0, 1, 2, 3, 2, 2, 2, 1, 1, 1, 0, 0, 3, "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", 2, "undo", 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 3, 2, 3, "undo", 1, 3, 2, "undo", 1, 2, 3, 3, 2, 3, 3, 1, "undo", "undo", "restart", 1, 1, 1, 1, 0, 1, 2, 3, 2, 2, 1, 1, 0, 3, 3, "undo", 2, 3, 3, 0, 3, "undo", 0, 3, 3, 2, 1, 2, 1, 0, 2, 1, 1, 0, 3, 3, 0, "undo", 0, 2, 1, 0, 0, 3, 3, "undo", 3, "undo", 0, 3, 3, 2, 3, 2, 1, 2, 1, 2, 1, 0, 0, 0, 1, 1, "undo", 1, 3, 1, 1, 2, 3, 2, 3, 2, 3, 0, 0, 0, 3, 3, 3, 3, 3, 2, 1], {file:"test_endstates/lightdown.txt"}, 13, "1645563041712.7258"]
	],



	[
		"I Need To Stop My Teleport Gun",
		[{file:"test_scripts/i_need_to_stop_my_teleport_gun.txt"}, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 1, 4, 4, 4, 4, 4, 4, 4, 4, 0, 3, 2, 4, 4, 1, 1, 3, 0, 3, 1, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, "undo", "undo", 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 2, 2, 3, 0, 2, 2, 3, 0, 3, 0, 2, 2, "undo", "undo", "undo", "undo", 3, 2, 3, 3, 0, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0], {file:"test_endstates/i_need_to_stop_my_teleport_gun.txt"}, 1, "1645578752808.7263"]
	],
	[
		"City road builder",
		[{file:"test_scripts/city_road_builder.txt"}, [0, 0, 0, 0, 3, 3, 3, 2, 2, 1, "undo", "undo", 3, 0, 0, 3, 3, 2, 2, 2, 3, 1, 0, 0, 0, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 3, 3, 1, 1, 2, 1, 1], {file:"test_endstates/city_road_builder.txt"}, 2, "1645569075149.173"]
	],
	[
		"Every Star Needs a Tree",
		[{file:"test_scripts/every_star_needs_a_tree.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", 1, "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", 0, "tick", "tick", "tick", 3, "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", 0, "tick", "tick", "tick", 3, "tick", 0, "tick", "tick", 1, "tick", "tick", "tick", "tick", 2, "tick", 1, "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", 1, "tick", 0, "tick", "tick", 3, "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", 0, "tick", "tick", "tick", 1, "tick", "tick", "tick", 3, "tick", "tick", 3, "tick", "tick", 2, "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", 2, "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "undo", "tick", "tick", "tick", "tick", "undo", "tick", "tick", "tick", "undo", "tick", "tick", "tick", "undo", "tick", "tick", "tick", "tick", "undo", "tick", "tick", "tick", "undo", "tick", "tick", "tick", "tick", "undo", "tick", "tick", "tick", "undo", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 2, "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", 1, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", 3, "tick", "tick", 2, "tick", 1, "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", 0, "tick", "tick", "tick", 1, "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", "tick", 2, "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 3, "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 2, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 3, "tick", "tick", 0, "tick", "tick", 0, "tick", 3, "tick", "tick", 0, "tick", "tick", 3, "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", "tick", "tick", 0, "tick", 1, "tick", "tick", "tick", 2, "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", 2, "tick", "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/every_star_needs_a_tree.txt"}, 0, "1645569315086.5925"]
	],
	[
		"Long Haul Space Flight",
		[{file:"test_scripts/long_haul_space_flight.txt"}, [3, 2, 1, 2, 3, 3, 3, 1, 0, 3, 0, 0, 2, 0, 3, 3, 3, 2, 2, 2, 3, 3, 0, 0, 0, 3, 2, 3, 3, 2, 3, 0, 0, 0, 2, "undo", 1, 2, 1, 1, 2, 2, 1, 1, 1, 0, 1, 0, 0, 0, 3, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 3, 2, 3, 2, 1, 0, 0, 0, 1, 3, 0], "background wall:0,0,0,0,0,0,0,0,0,0,background:1,\n1,1,1,1,0,0,0,1,background crate:2,1,1,\n1,0,0,0,1,1,1,1,1,0,0,\n0,1,0,0,0,1,0,0,1,background dthaw:3,1,\nbackground thaw:4,1,3,0,0,0,0,0,1,0,0,\n0,0,background target:5,1,1,1,1,background crate target:6,0,0,6,\nbackground crate defunct:7,1,background defunct:8,1,8,0,0,1,0,0,0,\n1,1,0,0,0,0,0,0,0,0,0,\n", 29, "1645570541337.319"]
	],
	[
		"Caramelban",
		[{file:"test_scripts/caramelban.txt"}, [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 2, 1, 3, 3, 2, 1, 0, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 0, 1, 2, 2, 2, 3, 3, 0, 0, 0, 0, 2, 1, 0, 0, 3, 3, 0, 3, 2, 1, 2, 3, 3, 3, 1, 1, 2, 1, 2, 2, 3, 3, 0, 0, 0, 3, 3, 3, 1, 1, 1, 0, 0, 0, 3, 2, 2, 0, 2, 2, 2, 1, 2, 2, 3, 0, 1, 0, 0, 3, 0, 3, 3, "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", "undo", 0, 1, 0, 0, 3, 3, 2, 1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 1, 0, 0, 1, 0, 3, 3, 1, 1, 1, 1, 1, 0, 0, 0, 3, 2, 3, "undo", 0, 3, 3, 2, 1, 3, 3, 3, 1, 2, 2, 3, 1, 2, 3, 1, 1, 1, 2, 0, 3, 3, 3, 3, 2, 2, 1, 0, 0, 0, 1, 0, 3, 3, 3, 3, 1, 1, 2, 1, 1, 1, 0, 3, 1, 1, 3, 3, 3, 1, 1, 0, 0, 3, 2, 0, 3, 2, 3, 3, 2, 3, 3, 3, 2, 0, 0, 1, 1, 0, 1, 0, 1, 1, 2, 2, 2, 3, 3, 0, 0, 1, 0, 1, 1, 3, 3, 3, 2, 3, 3, 3, 2, 0, 1, 1, 3, 3, 2], {file:"test_endstates/caramelban.txt"}, 1, "1645572641748.1404"]
	],
	[
		"Paralelism",
		[{file:"test_scripts/paralelism.txt"}, [3, 1, 0, 3, 2, 1, 2, 1, 1, 0, 0, 3, 1, 1, 1, 0, 1, 2, 3, 2, 1, 2, 2, 0, 3, 0, 1, 0, 3, 0, 3, 2, 1, 2, 1, 2, 0, 3, 0, 3, 0, 3, 2, 3, 0, 3, 1, 2, 3, 1, 0, 2, 1, 2], "background:0,activetile background ortplayer:1,0,0,0,0,0,activetile background:2,0,0,0,\n0,0,0,2,2,background void:3,2,3,0,0,0,\n0,3,2,3,0,0,3,0,0,0,2,\n0,3,3,0,3,0,3,2,3,3,0,\n2,0,2,0,3,3,3,3,0,3,3,\n3,0,0,0,2,0,0,0,0,0,0,\nbackground target:4,0,0,0,0,0,0,0,0,0,0,\n", 15, "1645572899761.3071"]
	],
	[
		"Psyshic push",
		[{file:"test_scripts/psyshic_push.txt"}, [2, 2, 2, 2, 3, 0, 0, 0, 1, 1, 0, 0, 3, 3, 1, 2, "undo", "undo", 2, 0, 3, 3, 2, 2, 1, 1, 3, 1, 2, 2, 2, 3, 1, 2, 3, 1, 1, 0, 0, 1, 3, 2, 0, 0, 3, 3, 1, 1, 2, 2, 2, 1, 1, 3, 3, 1, 1, 1, 1, 1, 0, 0, 3, 1, 2, 1, 0, 3], {file:"test_endstates/psyshic_push.txt"}, 32, "1645573610754.0828"]
	],
	[
		"I wanna change my position",
		[{file:"test_scripts/i_wanna_change_my_position.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", 0, "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", 4, "tick", "tick", "tick", "tick", "tick", "tick", 1, "tick", "tick", 3, "tick", 2, "tick", "tick", 4, "tick", 1, "tick", 0, "tick", "tick", 0, "tick", 3, "tick", "tick", 3, "tick", "tick", 2, "tick", "tick", "tick", 1, "tick", "tick", "tick", 2, "tick", "tick", 2, "tick", "tick", "tick", "tick", "tick", "tick", "tick", 4, "tick", 3, "tick", 3, "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", 4, "tick", "tick", "tick", 1, "tick", "tick", 1, "tick", 1, "tick", "tick", "tick", 0, "tick", "tick", 1, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], "background wall:0,0,0,0,0,background:1,1,1,\n0,background floor:2,2,2,0,0,0,1,\n0,background floor player target targetwithoutcrates:3,anim2 background crate floor memory:4,background floor target targetwithoutcrates:5,background crate floor:6,5,0,1,\n0,2,2,6,0,0,0,1,\n0,background crate floor target targetwithcrates:7,2,5,6,2,0,1,\n0,0,2,0,0,0,0,1,\n1,0,2,0,1,1,1,1,\n1,0,0,0,1,1,1,1,\n", 11, "1645574132967.735"]
	],
	[
		"Isle of Bloks",
		[{file:"test_scripts/isle_of_bloks.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 0, "tick", 1, "tick", "tick", 1, "tick", 0, "tick", 1, "tick", 1, "tick", 0, "tick", 3, "tick", "tick", "tick", "tick", "tick", 4, "tick", "tick", 4, 0, "tick", "tick", 3, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "undo", "tick", "tick", 2, "tick", "tick", 2, "tick", 3, "tick", "tick", "tick", 2, "tick", 3, "tick", 2, "tick", 1, "tick", "tick", 0, "tick", 3, "tick", "tick", 3, "tick", 3, "tick", "tick", 3, "tick", 3, "tick", "tick", 3, 2, "tick", "tick", 2, "tick", 1, "tick", 0, "tick", 3, "tick", 0, 1, "tick", "tick", 1, "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", "tick", 1, "tick", 0, "tick", 1, "tick", 2, "tick", "tick", 3, "tick", "tick", "tick", 1, "tick", 2, "tick", 3, "tick", 2, 1, "tick", "tick", 1, "tick", "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", 3, "tick", "tick", "tick", 0, "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick"], {file:"test_endstates/isle_of_bloks.txt"}, 11, "1645574325242.0908"]
	],
	[
		"SokokoS",
		[{file:"test_scripts/sokokos.txt"}, [3, 4, 0, 0, 0, 1, 1, 4, 3, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 3, 1, 2, 0, 3, 4, 0, 4, 1, 2, 0, 2, 3, 2, 2, 1, 1, 4, 0, 0, 0, 0, 1, 3, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 1, 0, 4, 1, 3, 3, 2, 4, 4, 0, 1, 1, 0, 0, 3, 4, 0, 0, 4, 1, 0, 3, 3, 2, 3, 2, 3, 3, 2, 2], {file:"test_endstates/sokokos.txt"}, 1, "1645642394866.7083"]
	],
	[
		"rigid disablement applies to whole rule-groups",
		[{file:"test_scripts/rigid_disablement_applies_to_whole_rule_groups.txt"}, [4], {file:"test_endstates/rigid_disablement_applies_to_whole_rule_groups.txt"}, 0, "1646700547379.7295"]
	],
	[
		"rigid applies to movements even if the movements were already present in the cell before rule application [i.e. if the rule doesn't do anything]",
		[{file:"test_scripts/rigid_applies_to_movements_even_if_the_movements_were_alread.txt"}, [4], {file:"test_endstates/rigid_applies_to_movements_even_if_the_movements_were_alread.txt"}, 0, "1646645466460.093"]
	],
	[
		"rigid applies to movements even if the movements are chagned by subsequent non-rigid rules in other groups",
		[{file:"test_scripts/rigid_applies_to_movements_even_if_the_movements_are_chagned.txt"}, [3], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,0,0,0,\n0,0,0,0,0,background player:2,0,2,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646647163086.0168"]
	],
	[
		"rigid applies to movements even if the objects are changed to different objects in the same layer",
		[{file:"test_scripts/rigid_applies_to_movements_even_if_the_objects_are_changed_t.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,0,0,0,\nbackground crate:2,0,2,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646647382066.4045"]
	],
	[
		"rigid applies to movements even if they are removed and subsequently readded",
		[{file:"test_scripts/rigid_applies_to_movements_even_if_they_are_removed_and_subs.txt"}, [3], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,0,0,0,\n0,0,0,0,0,background player:2,0,2,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646647562594.771"]
	],
	[
		"rigid applies to movements even if they are removed",
		[{file:"test_scripts/rigid_applies_to_movements_even_if_they_are_removed.txt"}, [3], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,0,0,0,\n0,0,0,0,0,background player:2,0,2,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646647631710.7463"]
	],
	[
		"rigid applies to movements even if the initial movement applied is an ACTION that is later changed to a movement",
		[{file:"test_scripts/rigid_applies_to_movements_even_if_the_initial_movement_appl.txt"}, [3], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,0,0,0,\nbackground crate:2,0,2,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\nbackground player:3,0,0,0,0,0,0,0,\n", 0, "1646647810474.9294"]
	],
	[
		"rigid applies to movements even if the initial movement applied is an STATIONARY that is later changed to a movement",
		[{file:"test_scripts/rigid_applies_to_movements_even_if_the_initial_movement_appl_2.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,0,0,background crate:2,\n0,2,0,0,0,0,0,0,\n0,0,0,background player:3,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646647871326.5627"]
	],
	[
		"rigid DOES NOT apply to objects if the rule is totally inert except for triggering SFX",
		[{file:"test_scripts/rigid_does_not_apply_to_objects_if_the_rule_is_totally_inert.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,background crate:2,0,0,\n2,0,0,0,0,0,0,0,\n0,0,0,background player:3,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646648625330.979"]
	],
	[
		"rigid DOES NOT apply to objects if a movement on the RHS is removed implicitly [by not mentioning a movement matched on the LHS]",
		[{file:"test_scripts/rigid_does_not_apply_to_objects_if_a_movement_on_the_rhs_is.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,0,0,0,\nbackground crate:2,2,0,0,0,0,0,0,\n0,0,0,background player:3,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646648958460.208"]
	],
	[
		"rigid DOES apply to objects if a movement on the RHS is removed explicitly [by not mentioning a movement matched on the LHS]",
		[{file:"test_scripts/rigid_does_apply_to_objects_if_a_movement_on_the_rhs_is_remo.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,0,0,background crate:2,\n0,2,0,0,0,0,0,0,\n0,0,0,background player:3,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646649072656.4766"]
	],
	[
		"rigid DOES apply to objects if a movement on the RHS is removed explicitly, even if the object is replaced by another one on the same layer",
		[{file:"test_scripts/rigid_does_apply_to_objects_if_a_movement_on_the_rhs_is_remo_2.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,background wall:1,0,0,0,background crate:2,\n0,2,0,0,0,0,0,0,\n0,0,0,background player:3,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646649296576.2854"]
	],
	[
		"rigid DOES NOT apply to objects if a movement on the RHS is assigned, and later the object is replaced by an object on another layer",
		[{file:"test_scripts/rigid_does_not_apply_to_objects_if_a_movement_on_the_rhs_is_2.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,background target wall:1,0,0,0,0,\nbackground target:2,background crate:3,0,0,0,0,0,0,\n0,0,0,background player:4,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646649418277.2751"]
	],
	[
		"misc rigid test",
		[{file:"test_scripts/misc_rigid_test.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\nbackground wall:1,0,background crate:2,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,background player:3,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646698655056.5757"]
	],
	[
		"rigid: even removing and re-adding an object doesn't remove the rigid tag from that position and layer.",
		[{file:"test_scripts/rigid_even_removing_and_re_adding_an_object_doesn_t_remove_t.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,background wall:1,background crate:2,\n0,2,0,0,0,0,0,0,\n0,0,0,0,0,0,background player:3,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646698763919.0374"]
	],
	[
		"rigid: even a rigid rule removing an object marks the position/layer of the removed object as rigid.",
		[{file:"test_scripts/rigid_even_a_rigid_rule_removing_an_object_marks_the_positio.txt"}, [4], "background:0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,background crate:1,0,0,0,background wall:2,0,\n0,0,0,0,0,0,1,0,\n0,0,0,0,0,0,background player:3,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1646699000524.8464"]
	],
	[
		"One player, unlimited rigidbodies",
		[{file:"test_scripts/one_player_unlimited_rigidbodies.txt"}, [1, 1, 2, 3, 3, 2, 3, 2, 3, 3, 0, 0, 0, 0, 0, 3, 0, 1, 1, 1, 1, 2, 1, 1, 0, 0, 3, 0, 3, 2, 3, 3, 3], {file:"test_endstates/one_player_unlimited_rigidbodies.txt"}, 0, "1646701772316.744"]
	],
	[
		"Many parallel players, unlimited rigidbodies",
		[{file:"test_scripts/many_parallel_players_unlimited_rigidbodies.txt"}, [3, 3, 3, 0, 3, 3, 2, 2, 2, 1, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 0, 1, 1, 1, 1, 0, 3, 1, 1, 1, 3, 3, 3, 1, 0, 1, 1, 1], {file:"test_endstates/many_parallel_players_unlimited_rigidbodies.txt"}, 0, "1646703814545.0725"]
	],
	[
		"Restart command on level start",
		[{file:"test_scripts/restart_command_on_level_start.txt"}, [2, 1, 0], "background wall:0,0,0,0,0,0,\n0,0,background:1,background crate:2,background player target:3,1,\n1,0,0,background target:4,1,1,\n1,1,0,0,0,0,\n1,2,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1646748136288.8362"]
	],
	[
		"startloop stoploop vortex check",
		[{file:"test_scripts/startloop_stoploop_vortex_check.txt"}, [0, 1, 2, 3, 2, 2, 3, 1, 1, 2, 3], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,\n1,0,0,background target:3,1,1,\n1,background player:4,0,0,0,0,\n1,background crate:5,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1646748363863.9624"]
	],
	[
		"Audio Test 1",
		[{file:"test_scripts/audio_test_1.txt"}, [2, 1, 0, 3, 3, 3, 2, 1, 0, 1, 2, 2], "background wall:0,0,0,0,0,0,\n0,0,background:1,background crate:2,background target:3,1,\n1,0,0,3,1,1,\nbackground player:4,2,0,0,0,0,\n1,1,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", "0", "1646875706306.7224", ["36772507", "36772507", "36772507", "34411705"]]
	],
	[
		"Audio Test 2",
		[{file:"test_scripts/audio_test_2.txt"}, [0, 1, 2, 2, 2], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background target:2,background player:3,\nbackground crate:4,0,0,2,1,1,\n1,1,0,0,0,0,\n1,4,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", "0", "1646876897098.8142", ["5", "5", "6"]]
	],
	[
		"Audio Test 3",
		[{file:"test_scripts/audio_test_3.txt"}, [2, 1, 0, 1, 4, 0, 3, 2, 2, 4, 3, 0, 1, 1, 1], "background wall:0,0,0,background crate:1,0,0,\n0,0,background:2,2,background crate target:3,2,\n2,0,0,background target:4,2,background player:5,\n2,2,0,0,0,0,\n2,2,0,0,2,2,\n0,2,2,0,2,2,\n2,0,0,0,0,2,\n", 0, "1646877211807.3489", ["5", "5", "1", "5", "6", "3", "4", "5", "5", "1", "5", "1", "5", "3", "4", "5", "5", "1", "5", "2", "6", "2", "6"]]
	],
	[
		"aggregates in win conditions 1",
		[{file:"test_scripts/aggregates_in_win_conditions_1.txt"}, [0, 1, 2, 3, 3, 2, 3, 0, 0, 3, 2, 1, 2, 1, 1], "background:0,background target:1,0,0,\n0,0,0,0,\n", 1, "1646902744672.2258", ["36772507", "36772507"]]
	],
	[
		"aggregates in win conditions 2",
		[{file:"test_scripts/aggregates_in_win_conditions_2.txt"}, [3, 3, 3, 0, 3, 2], "background:0,background target:1,0,0,\n0,0,0,0,\n", 0, "1646902824835.6414", ["36772507"]]
	],
	[
		"aggregates in win conditions 3",
		[{file:"test_scripts/aggregates_in_win_conditions_3.txt"}, [0, 1, 2, 3, 3, 2, 2, 1, 1, 0, 2, 3, 0, 3, 0, 1], "background:0,background target:1,0,0,\n0,0,0,0,\n", 1, "1646903196185.7786", ["36772507", "36772507"]]
	],
	[
		"aggregates in win conditions 4",
		[{file:"test_scripts/aggregates_in_win_conditions_4.txt"}, [3, 2, 1, 2, 3, 3, 0, 1, 0, 0, 3, 3, 3, 2, 2, 1, 1, 1], "background:0,background target:1,0,0,\n0,0,0,0,\n", 0, "1646903268108.808", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],
	[
		"aggregates in win conditions 5",
		[{file:"test_scripts/aggregates_in_win_conditions_5.txt"}, [3, 3, 3, 2, 1, 1, 0, 3, 3, "undo", "undo", "undo", "undo", "undo", 0, 3, 3, 2, 1, 3, 0, 3, 3, 2, 1, 1, 0, 1, 1, 1, 3, 2, "undo", 1, 2, 0, 3, 2, 0, 3, 2], "background player:0,background:1,1,\nbackground cratedot:2,1,1,\nbackground crate target:3,1,1,\n", 2, "1646903377668.326", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],
	[
		"aggregates in win conditions 6",
		[{file:"test_scripts/aggregates_in_win_conditions_6.txt"}, [1, 1, 2, 2, 2, 2, 3, 3, 3, 0, 3, 0, 0, 0, 1, 2, 3, 2, 2, 2, 1, 1, 0, 2, 1, 0, 3, 2, 3, 2, 2, 2, 1, 0, 2, 1, 1, 3, 0, 1, 3, 2, 2, 3, 2, 2, 1, 0, 2, 1, 1, 0, 3, 2, 3, 3, 0, 0, 1, 3, 2, 2, 1, 0], "background player:0,background:1,1,\nbackground cratedot:2,1,1,\nbackground crate target:3,1,1,\n", 0, "1646903481599.992", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],
	[
		"aggregates in win conditions 7",
		[{file:"test_scripts/aggregates_in_win_conditions_7.txt"}, [3, 2, 3, 2, 2, 2, 1, 0, 2, 1, 1, 0, 2, 3, 3, 3, 0, 0, 1, 2, 2, 1, 1, 0, 3, 2, 3, 0, 3, 3, 2, 2, 2, 2, 1, 0, 2, 2, 1, 1, 0, 2, 3, 3, 1, 0, 3, 3, 0, 0, 1, 2, 0, 0, 1, 1, 2, 3, 0, 3, 2, 3, 2, 2, 2, 1, 0], "background:0,0,0,0,0,\n0,0,background target:1,background crate cratedot:2,0,\nbackground player:3,0,1,2,0,\n0,background cratedot:4,1,background crate:5,0,\n0,0,0,0,0,\n", 0, "1646903516649.6477", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],
	[
		"double ellipsis simple test",
		[{file:"test_scripts/double_ellipsis_simple_test.txt"}, [3], "background:0,0,0,0,background player:1,1,1,1,1,1,1,1,\n1,1,1,1,background crate:2,2,0,0,2,2,2,2,\n0,background target:3,2,2,3,3,3,0,3,3,3,3,\n3,3,3,3,3,0,3,3,0,0,0,3,\n", 0, "1647256697085.1719", ["36772507"]]
	],
	[
		"double ellipsis simple test #2",
		[{file:"test_scripts/double_ellipsis_simple_test_2.txt"}, [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 0, 1, 2, 1, 1, 1, 0, 1, 2, 1, 1, 1, 3, 3, 3, 3, 2, 1, 1, 1, 1, 1, "undo", 1, 3, 3, 3, 3, 3, 3, 2, 0, 0, 1, 2, 2, 1, 0, 3, 2, 1, 0, 1, 2, 1, 1, 1, "undo", 1, 1, 3, 3, 3, 3, 3, 3, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, "undo", 1, "undo"], "background:0,background target:1,1,1,0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,background crate target:2,\n2,1,1,0,0,background crate:3,3,3,2,0,0,3,3,3,3,0,0,3,3,3,\n3,0,0,0,0,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,background player:4,0,0,0,0,0,4,0,0,0,0,0,4,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\n", 0, "1647256758831.4763", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],
	[
		"double ellipsis simple test #3",
		[{file:"test_scripts/double_ellipsis_simple_test_3.txt"}, [2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 3, 1, 0, "undo", 0, "undo", 1, 0, 1, 1, 0, 3, 3, 3, 0, "undo", 0, "undo", 0, 2, 2, 2, 3, 3, 3, 3, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 2], {file:"test_endstates/double_ellipsis_simple_test_3.txt"}, 0, "1647256841954.48", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],
	[
		"double ellipsis simple test #4",
		[{file:"test_scripts/double_ellipsis_simple_test_4.txt"}, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, 1, 1, 1, 0, 3, "undo", "undo", "undo", 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 0, 3, 2, 0, 2, 0, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, "undo", 0, 0], {file:"test_endstates/double_ellipsis_simple_test_4.txt"}, 0, "1647256905962.6406", ["36772507", "36772507", "36772507"]]
	],
	[
		"double ellipsis simple test #5",
		[{file:"test_scripts/double_ellipsis_simple_test_5.txt"}, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 0, "undo", 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, "undo", 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 3, 3, 3, 3, 2], {file:"test_endstates/double_ellipsis_simple_test_5.txt"}, 0, "1647256954689.7512", ["36772507", "36772507", "36772507", "36772507", "36772507"]]
	],

	[
		"Double-ellipsis testcase",
		[{file:"test_scripts/double_ellipsis_testcase.txt"}, [3], "background:0,0,0,0,0,0,0,0,0,background d:1,0,\n1,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,b background:2,0,0,0,0,background player:3,0,\n0,0,0,0,0,0,0,0,background c:4,0,0,\n0,2,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,4,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,\n", 0, "1647258256850.6755", []]
	],
	[
		"Double-ellipsis delicate test case",
		[{file:"test_scripts/double_ellipsis_delicate_test_case.txt"}, [3], "background:0,0,0,0,0,0,0,0,0,background d:1,0,\n0,1,0,0,0,0,0,0,0,0,0,\n0,0,background player:2,0,b background:3,0,0,0,0,0,0,\n0,0,0,0,0,0,0,background c:4,0,0,0,\n0,3,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,4,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,\n", 0, "1647258285625.6372", []]
	],
	[
		"Autowin",
		["title Autowin\nauthor lazugod\n\nnoaction\nrequire_player_movement\n\n========\nOBJECTS\n========\n\nBackground .\nblack\n\nPlayer p\nwhite\n.....\n.000.\n.000.\n00000\n.0.0.\n\nCrate c\norange\n00000\n0...0\n0...0\n0...0\n00000\n\nWall #\nbrown\n\n=======\nLEGEND\n=======\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate, Wall\n\n======\nRULES\n====== \n\n[ > Player | Crate ] -> [ > Player | > Crate ]\n[ > Player | Crate | no Crate no Wall ] -> Cancel Message Canceled push message\n\n==============\nWINCONDITIONS\n==============\n\nno Player\nno Wall\nno Crate\n\n=======\nLEVELS\n=======\n\n#####\n#.cp#\n#####\n\nMessage Should-be-unreachable victory\n", [1, 1], "background wall:0,0,0,0,background:1,\n0,0,background crate:2,0,0,\nbackground player:3,0,0,0,0,\n", 0, "1708473053336.5632", []]
	],
	[
		"Autowin2",
		["title Autowin\nauthor lazugod\n\nnoaction\nrequire_player_movement\n\n========\nOBJECTS\n========\n\nBackground .\nblack\n\nPlayer p\nwhite\n.....\n.000.\n.000.\n00000\n.0.0.\n\nCrate c\norange\n00000\n0...0\n0...0\n0...0\n00000\n\nWall #\nbrown\n\n=======\nLEGEND\n=======\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer, Crate, Wall\n\n======\nRULES\n====== \n\n[ > Player | Crate ] -> [ > Player | > Crate ]\n[ > Player | Crate | no Crate no Wall ] -> Cancel Message Canceled push message\n\n==============\nWINCONDITIONS\n==============\n\nno Player\nno Wall\nno Crate\n\n=======\nLEVELS\n=======\n\n#####\n#.cp#\n#####\n\nMessage Should-be-unreachable victory\n", [1, 1], "background wall:0,0,0,0,background:1,\n0,0,background crate:2,0,0,\nbackground player:3,0,0,0,0,\n", 0, "1708473053336.5632", []]
	],
	[
		"Head Skuller",
		[{file:"test_scripts/head_skuller.txt"}, [3, 2, 2, 2, 1, 0, 2, 3, 3, 0, 2, 3, 3, 3, 0, 2, 1, 0, 2, 3, 3, 0, 2, 3, 3, 0, 2, 3, 3, 0, 1, 2, 2, 2, 2, 3, 0, 2, 1, 1, 0, 2, 1, 1, 0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 3, 2, 2, 2, 1, 0, 2, 3, 3, 0, 2, 3, 3, 0, 2, 3, 3, 0, 2, 3, 3, 0, 2, 3, 3, 0, 1, 2, 2, 2, 2, 3, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 3, 2, 2, 2, 1, 0, 2, 3, 3, 0, 2, 3, 3, 0, 2, 3, 3, 0, 2, 3, 3, 0, 2, 3, 3, 0, 2, 1, 2, 2, 2, 3, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0, 3, 0, 0, 0, 1, 2, 0, 3, 3, 2, 0, 3, 3, 2, 0, 3, 3, 2, 0, 3, 3, 2, 0, 3, 3, 2, 1, 0, 0, 0, 0, 3, 2, 0, 1, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 0, 1, 1, 2, 1, 0, 1, 2, 3, 0, 0, 0, 0, 1, 2, 3, 3, 2, 3, 0, 0, 3, 2, 0, 3, 3, 2, 0, 3, 3, 2, 0, 3, 3, 2, 1, 0, 0, 0, 0, 3, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 1, 1, 2, 0, 3, 0, 0, 0, 0, 1, 2, 0, 3, 3, 2, 0, 3, 3, 2, 0, 3, 3, 2, 0, 3, 3, 2, 0, 3, 3, 0, 2, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1], {file:"test_endstates/head_skuller.txt"}, 1, "1713895320743.1035", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],
	[
		"Sokoban... in 3D!",
		[{file:"test_scripts/sokoban_in_3d.txt"}, [0, 0, 0, 0, 3, 2, 3, 3, 1, 2, 2, 1, 2, 2, 3, 3, 3, 0, 1, 2, 1, 1, 1, 1, 1, 0, 0, 3, 3, 3, 2, 3, 0, 2, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 3, 3, 2, 2, 1, 0, 1, 2, 2, 2, 3, 2, 2, 1, 1, 1, 3], {file:"test_endstates/sokoban_in_3d.txt"}, 4, "1714157696832.4265", []]
	],

	[
		"single screen scream",
		[{file:"test_scripts/single_screen_scream.txt"}, [2, 3, 2, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 0, 1, 0, 3, 3, "undo"], {file:"test_endstates/single_screen_scream.txt"}, 0, "1714157756267.3792", []]
	],

	[
		"Crate Assembler",
		[{file:"test_scripts/crate_assembler.txt"}, [3, 3, 2, 1, 0, 1, 1, 2, 3, 0, 0, 0, 1, 2, 0, 0, 3, 2, 2, 3, 2, 1, 1, 2, 1, 0], {file:"test_endstates/crate_assembler.txt"}, 29, "1714157781825.6523", []]
	],

	[
		"Clearing space",
		[{file:"test_scripts/clearing_space.txt"}, [3, 0, 0, 2, 2, 4, 0, 0, 0, 4, 0, 0, 1, 2, 4, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 1, 0, 0, 3, 3, 3, 3, 2, 2, 2, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 1, 0, 1, 1, 0, 3, 0, 0, 0, 1, 1, 1, 2, 2, 1], {file:"test_endstates/clearing_space.txt"}, 11, "1714157801411.8848", []]
	],

	[
		"Mini Jill Off",
		[{file:"test_scripts/mini_jill_off.txt"}, ["tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", "tick", 3, "tick", 3, "tick", 3, "tick", 3, "tick", 0, "tick", 3, "tick", 0, 1, 1, "tick", "tick", "tick", 0, 3, "tick", "tick", "tick", 3, "tick", "tick", "tick", 0, "tick", 0, "tick", 3, "tick", 0, 1, "tick", "tick", 1, "tick", 1, "tick", "tick", 1, "tick", "tick", "tick", 1, "tick", "tick", 1, "tick", "tick", 1, "tick", "tick"], {file:"test_endstates/mini_jill_off.txt"}, 4, "1714157849435.0376", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],

	[
		"Big dog and little dog",
		[{file:"test_scripts/big_dog_and_little_dog.txt"}, [3, 2, 2, 2, 1, 1, 1, 2, 1, 0, 0, 0, 0, 2, 2, 3, 3, 0, 1, 1, 2, 1, 3, 0, 1, 1, 2, 2, 1, 0, 0, 1, 0, 0, 0, 3, 3], {file:"test_endstates/big_dog_and_little_dog.txt"}, 11, "1714158104939.6763", ["98218307", "98218307", "98218307", "98218307", "98218307", "98218307", "98218307", "98218307", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98923704", "98218307", "98218307", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "98218307", "98923704", "2421301"]]
	],

	[
		"The Monsterous Autoshove",
		[{file:"test_scripts/the_monsterous_autoshove.txt"}, [3, 3, 0, 2, 3, 3, 0, 1, 3, 3, 0, 0, 0, 1, 1, 1, 2, 1, 2, 1, 2, 3, 1, 0, 3, 3, 3, 1, 1, 2, 3, 3, 2, 3, 3, 0, 0, 2, 2, 1, 1, 0, 0, 2, 1, 1, 0, 0, 0, 3, 0, 2, 3, 3, 2, 0, 0, 0, 1, 2, 1, 1], "background wall:0,0,0,0,0,0,0,0,0,\n0,0,0,background target:1,background:2,2,2,1,2,\n2,2,1,0,0,2,2,background one_neighbour:3,2,\n2,2,2,2,2,0,0,2,background crate:4,\n4,2,2,2,2,2,2,0,0,\n3,4,background player:5,2,3,2,2,2,2,\n0,0,2,2,2,3,4,2,2,\n2,2,0,0,2,2,2,2,2,\n4,3,2,2,0,0,1,2,2,\n3,background crate target:6,2,2,2,1,0,0,0,\n0,0,0,0,0,0,0,0,0,\n", 9, "1714158135585.5205", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],

	[
		"ecosystem generator",
		[{file:"test_scripts/ecosystem_generator.txt"}, [3, 3, 0, 0, 1, 2, 2, 0, 0, 3, 3, 3, 3, 2, 2, 3, 2, 1, 1, 1, 3, 0, 0, 0, 1, 0, 0, 3, 2, 2, 2, 2, 3, 2, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 2, 1, 2, 3, 3, 3, 3, 3, 3, 0, 3, 2, 2, 2, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0, 1, 1, 2, 0, 3, 2, 3, 2, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 2, 2, 1, 0, 1, 0, 3, 3, 3, 3, 3, 3, 0, 3, 2, 0, 0, 3, 2, 2, 3, 3, 2, 2, 1, 0, 0, 3, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 2, 1, 2, 3, 0, 3, 3, 0, 3, 3, 2, 1, 1, 1, 0, 1, 2, 0, 3, 3, 3, 3, 3, 2, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3, 0, 0, 3, 3, 0, 0, 3, 2, 2], {file:"test_endstates/ecosystem_generator.txt"}, 1, "1714158163344.676", ["82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307", "82819307"]]
	],

	[
		"B*TTEATER",
		[{file:"test_scripts/b_tteater.txt"}, [0, 3, 0, 0, 0, 1, 1, 2, 3, 2, 1, 2, 2, 1, 1, 0, 1, 0, 1, 0, 3, 0, 3, 3, 3, 2, 1, 1, 2, 1, 2, 3, 3, 2, 3], {file:"test_endstates/b_tteater.txt"}, 3, "1714158223251.8372", []]
	],

	[
		"Fickle Fred",
		[{file:"test_scripts/fickle_fred.txt"}, [0, 0, 3, 3, 0, 0, 1, 0, 3, 2, 2, 2, 2, 1, 0, 0, 2], "background wall:0,0,0,0,0,0,0,\n0,0,background:1,1,1,1,1,\n1,0,0,1,background target:2,background crate:3,2,\n3,1,0,0,1,3,background crate target:4,\n1,4,1,0,0,1,2,\n3,background playerpush target:5,1,1,0,0,1,\n1,1,1,1,1,0,0,\n0,0,0,0,0,0,0,\n", 7, "1714158245818.2236", ["36772507", "36772507", "36772507", "36772507"]]
	],

	[
		"WITCH LIFTER",
		[{file:"test_scripts/witch_lifter.txt"}, [3, 3, 3, 0, 3, 3, 2, 3, 3, 0, 3, 3, 3, 2, 3, 3, 3, 1, 3, 1], {file:"test_endstates/witch_lifter.txt"}, 13, "1714158260875.6018", ["7670907", "7670907", "7670907", "44641307", "63010507", "40549507", "7670907", "44641307", "7670907", "44641307", "40549507", "7670907", "7670907", "44641307", "63010507", "40549507", "44641307", "7670907", "44641307", "7670907", "44641307", "7670907", "40549507", "44641307", "7670907", "63010507"]]
	],

	[
		"The Red Ring of Immortality",
		[{file:"test_scripts/the_red_ring_of_immortality.txt"}, [1, 0, 0, 0, 0, 0, 3, 0, 3, 1, 2, 2, 1, 0, 3, 3, 2, 2, 3, 0, 2, 2, 1, 0, 0], "background:0,background fire:1,0,0,0,background move_piece:2,0,0,\n0,0,1,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n2,0,0,2,0,background wall:3,1,0,\n0,background player player_move_left:4,1,3,0,0,3,1,\n0,1,0,background target:5,3,3,0,0,\n3,0,0,2,0,0,0,0,\n0,0,0,0,0,3,0,0,\n0,0,3,2,0,0,0,3,\n", 7, "1714158279281.9524", []]
	],
	[
		"cover 'random' bit of getPropertiesFromCell",
		[{file:"test_scripts/cover_random_bit_of_getpropertiesfromcell.txt"}, [1, 0, 3, 2, 2, 3, 0, "restart", 1, 0, 3, 2, 2, 2, 3, 0, 3, "undo", 3, 0, 3, 3, 2, 1, 0, 1, 2, 1], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background target:2,1,\n1,0,0,2,1,1,\nbackground player:3,1,0,0,0,0,\n1,2,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1714412458646.9324", []]
	],
	[
		"coverage of concretizeMovingInCellByAmbiguousMovementName",
		[{file:"test_scripts/coverage_of_concretizemovingincellbyambiguousmovementname.txt"}, [0, 2, 3, 2, 0, 3, 3, 2, 0, 3, 0, 1, 3], "background wall:0,0,0,0,0,0,\n0,0,background:1,background player:2,background target:3,1,\n1,0,0,3,1,background crate:4,\n1,1,0,0,0,0,\n1,4,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1714412623626.8494", ["36772507"]]
	],
	[
		"trigger delete newmetadata.flickscreen",
		[{file:"test_scripts/trigger_delete_newmetadata_flickscreen.txt"}, [0, 1, 2], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background player target:2,background crate:3,\n1,0,0,background target:4,1,1,\n1,1,0,0,0,0,\n1,3,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1714413486856.004", ["36772507"]]
	],
	[
		"trigger delete newmetadata.zoomscreen",
		[{file:"test_scripts/trigger_delete_newmetadata_zoomscreen.txt"}, [2, 3], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate target:2,1,\n1,0,0,background target:3,1,1,\n1,1,0,0,0,0,\n1,background player:4,0,0,1,1,\n0,1,background crate:5,0,1,1,\n1,0,0,0,0,1,\n", 0, "1714413369887.9194", ["36772507"]]
	],
	[
		"parser rigid in strange place highlighting test",
		[{file:"test_scripts/parser_rigid_in_strange_place_highlighting_test.txt"}, [0, 2, 1, 0, 0, 1, 2, 2], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background player target:2,background crate:3,\n1,0,0,background target:4,1,1,\n1,1,0,0,0,0,\n1,3,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1714418847274.8472", ["36772507"]]
	],
	[
		"background_color transparent code path",
		[{file:"test_scripts/background_color_transparent_code_path.txt"}, [0, 1, 2], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background player target:2,background crate:3,\n1,0,0,background target:4,1,1,\n1,1,0,0,0,0,\n1,3,0,0,1,1,\n0,1,1,0,1,1,\n1,0,0,0,0,1,\n", 0, "1714419533813.432", ["36772507"]]
	],
	[
		"Levels can not contain glyphs that resemble section names #976",
		[{file:"test_scripts/levels_can_not_contain_glyphs_that_resemble_section_names_97.txt"}, [0, 3, 2, 3, 3], "background:0,0,0,0,0,\n0,background s:1,1,0,background player:2,\n", 1, "1733177473747.7947", []]
	],
	[
		"Weirdness with = as glyph",
		[{file:"test_scripts/weirdness_with_as_glyph.txt"}, [2, 1, 0, 3, 3, 3, 2, 1, 0, 1, 1, 2, 2, 3, 0, 1, 0, 3, 0, 0, 1, 2, 3, 2, 2, 3, 3, 0, 1, 2, 1, 0, 0, 3, 2, 2, 1, 0, 1], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,background crate:2,1,\n1,0,0,1,0,background player:3,\nbackground crate target:4,1,0,0,1,2,\nbackground target:5,5,1,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", 0, "1733826211976.6633", ["36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507", "36772507"]]
	],
	[
		"Missing/Skipping Rules? Objects disappear for no reason? #1046",
		["title Missing/Skipping Rules? Objects disappear for no reason? #1046\nauthor Tyler Johnson\n\ndebug\nverbose_logging\nagain_interval 0.1\n\n\n========\nOBJECTS\n========\n\nBackground .\ngray\n\n\nPlayer\nblack orange blue #053A8E\n.000.\n.111.\n22222\n.333.\n.3.3.\n\nMark\ngreen\n.....\n.....\n.....\n.....\n0....\n\n\n\n\n=======\nLEGEND\n=======\n\np = Player and Mark\n\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nPlayer\nMark\n\n======\nRULES\n======\n\n\n[ Player no Mark ] -> [ ] restart\n\n[ Mark ] -> [] again\n\n\n==============\nWINCONDITIONS\n==============\n\n\n=======\nLEVELS\n=======\n\np\n\n", [3, 0, 1, 2], "background mark player:0,\n", 0, "1742144828942.1826", []]
	],
	[
		"aggregate player allowed #1032",
		[{file:"test_scripts/aggregate_player_allowed_1032.txt"}, [3, 3, 2, 2, 1, 1, 1, 3, 3, 0, 1, 1, 2, 3], "background:0,0,background player_body:1,background hat wall:2,\n0,0,0,0,\n0,background target:3,0,0,\n0,background hat player_body:4,3,0,\n0,0,0,0,\n", 0, "1742491557150.5847", []]
	],
	[
		"aggregate player allowed B #1032",
		[{file:"test_scripts/aggregate_player_allowed_b_1032.txt"}, [1, 1, 2, 2], "background:0,background player_body:1,background wall:2,0,\n0,0,0,background hat target:3,\n0,0,0,background target:4,\n0,0,0,0,\n", 1, "1742491588699.806", []]
	],
	[
		"aggregate player allowed C #1032",
		[{file:"test_scripts/aggregate_player_allowed_1032.txt"}, [1, 2, 2], "background:0,0,background hat player_body:1,background wall:2,\n0,0,0,background player_body:3,\n0,background target:4,0,0,\nbackground hat:5,0,4,0,\n0,0,0,0,\n", 0, "1742491611738.6934", []]
	],
	[
		"aggregate player allowed D #1032",
		[{file:"test_scripts/aggregate_player_allowed_1032.txt"}, [2, 3, 3, 2, 1, 1, 2, 2, 1], "background wall:0,0,0,0,0,0,\n0,0,background:1,1,1,1,\n1,0,0,1,0,background crate:2,\nbackground target:3,1,0,0,1,background player_body:4,\nbackground crate target:5,5,1,0,0,1,\n1,1,1,1,0,0,\n0,0,0,0,0,0,\n", 0, "1742491637124.892", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 1",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_1.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_1.txt"}, 0, "1742756593518.8354", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 2",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_2.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_2.txt"}, 0, "1742756609815.8584", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 3",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_3.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_3.txt"}, 0, "1742756629113.536", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 4",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_4.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_2.txt"}, 0, "1742756654284.0566", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 5",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_5.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_3.txt"}, 0, "1742756674648.0125", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 6",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_6.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_6.txt"}, 0, "1742756693770.5415", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 7",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_7.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_3.txt"}, 0, "1742756711510.7703", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 8",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_8.txt"}, [4], "background:0,0,background target:1,background l4c l5a:2,background player target2:3,1,background l4a:4,background l4b:5,\nbackground l4c:6,background l5a:7,background l5b:8,background l5c:9,0,0,background player target:10,1,\n1,3,background target2:11,2,11,11,background crate2:12,background player:13,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,13,0,\n0,0,0,0,0,0,0,0,\n", 0, "1742756728693.6768", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 9",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_9.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_9.txt"}, 0, "1742756744663.015", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 10",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_10.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_2.txt"}, 0, "1742756759082.899", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 11",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_11.txt"}, [4], "background:0,0,background target:1,background l4c l5a:2,background target2:3,background crate target:4,background l4a:5,background l4b:6,\nbackground l4c:7,background l5a:8,background l5b:9,background l5c:10,0,0,1,1,\n4,3,3,2,background crate target2:11,3,background crate2:12,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1742756820516.7627", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 12",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_12.txt"}, [4], "background:0,0,background target:1,background l4c l5a:2,background target2:3,background crate target:4,background l4a:5,background l4b:6,\nbackground l4c:7,background l5a:8,background l5b:9,background l5c:10,0,0,background player target:11,background target wall:12,\n4,3,3,2,background crate target2:13,3,background crate2:14,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1742756840269.2712", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 13",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_13.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_6.txt"}, 0, "1742756855203.018", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 14",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_14.txt"}, [4], "background:0,background wall:1,0,background l4c l5a:2,background player target2:3,0,background l4a:4,background l4b:5,\nbackground l4c:6,background l5a:7,background l5b:8,background l5c:9,0,0,background player target:10,1,\n0,3,background target2 wall:11,2,background target2:12,12,background crate2:13,background player:14,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,14,0,\n0,0,0,0,0,0,0,0,\n", 0, "1742756871876.9214", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 15",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_15.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_15.txt"}, 0, "1742756893786.2488", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 16",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_16.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_16.txt"}, 0, "1742756911727.6157", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 17",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_17.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_17.txt"}, 0, "1742756925012.9663", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 18",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_18.txt"}, [4], "background:0,background wall:1,0,background l4c l5a:2,background player target2:3,0,background l4a:4,background l4b:5,\nbackground l4c:6,background l5a:7,background l5b:8,background l5c:9,0,0,background player:10,1,\n0,3,background target2 wall:11,2,background crate target2:12,background target2:13,background crate2:14,10,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,10,0,\n0,0,0,0,0,0,0,0,\n", 0, "1742756940001.4253", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 19",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_19.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_17.txt"}, 0, "1742756950813.8457", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 20",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_20.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_20.txt"}, 0, "1742756964062.3767", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 21",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_21.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_20.txt"}, 0, "1742756976220.8772", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 22",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_22.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_20.txt"}, 0, "1742757040764.0063", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 23",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_23.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_3.txt"}, 0, "1742757050710.4363", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 24",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_24.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_3.txt"}, 0, "1742757077541.41", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 25",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_25.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_20.txt"}, 0, "1742757091293.6487", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 26",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_26.txt"}, [4], "background:0,background wall:1,background target:2,background l4c l5a:3,background target2:4,2,background l4a:5,background l4b:6,\nbackground l4c:7,background l5a:8,background l5b:9,background l5c:10,0,0,2,background target wall:11,\n2,4,background target2 wall:12,3,4,4,background crate2:13,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,\n", 0, "1742757107458.011", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 27",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_27.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_2.txt"}, 0, "1742757118828.476", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 28",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_28.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_6.txt"}, 0, "1742757135302.369", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 29",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_29.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_17.txt"}, 0, "1742757148460.7327", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 30",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_30.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_30.txt"}, 0, "1742757335371.5312", []]
	],

	[
		"#1067 [ player no wall ] test suite, No. 31",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_31.txt"}, [4], {file:"test_endstates/1067_player_no_wall_test_suite_no_31.txt"}, 0, "1742757371757.3948", []]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 32",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_32.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_32.txt"},0,"1742757396821.3445",[]]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 33",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_33.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_33.txt"},0,"1742757418803.0024",[]]
	],


	[
		"#1067 [ player no wall ] test suite, No. 34",
		[{file:"test_scripts/1067_player_no_wall_test_suite_no_34.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_30.txt"},0,"1742757433834.3035",[]]
		],
		
[
	"#1067 [ player no wall ] test suite, No. 35",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_35.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_35.txt"},0,"1742757450732.3696",[]]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 36",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_36.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_36.txt"},0,"1742757469739.958",[]]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 37",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_37.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_35.txt"},0,"1742757481133.478",[]]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 38",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_38.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_38.txt"},0,"1742757492740.5579",[]]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 39",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_39.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_36.txt"},0,"1742757511437.6326",[]]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 40",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_40.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_40.txt"},0,"1742757525874.0637",[]]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 41",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_41.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_35.txt"},0,"1742757544433.4973",[]]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 42",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_42.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_42.txt"},0,"1742757556240.4824",[]]
	],
	
[
	"#1067 [ player no wall ] test suite, No. 43",
	[{file:"test_scripts/1067_player_no_wall_test_suite_no_43.txt"},[4],{file:"test_endstates/1067_player_no_wall_test_suite_no_43.txt"},0,"1742757569325.104",[]]
	],	
	[
	"#1094 winning broken in Cratopia",
	[{file:"test_scripts/1094_winning_broken_in_cratopia.txt"},[1,1,3,3,1],"bluesky:0,bluesky grass:1,0,1,0,1,0,1,\n0,1,0,1,bluesky playerleft:2,1,0,1,\n",0,"1743014143027.1116",[]]
	],
	[
	"lhsHasOverlappingRequiredLayers",
	["title My Game\nauthor My Name Here\nhomepage www.puzzlescript.net\n\nverbose_logging\ndebug\n\n========\nOBJECTS\n========\n\nBackground\ngreen\n\nTarget\ndarkblue\n\nWall\nbrown\n\nPlayer\nblue\n\nCrate\norange\n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\n* = Crate\n@ = Crate and Target\ncat = Crate or Target\nO = Target\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nTarget\nPlayer, Wall, Crate\n\n======\nRULES\n======\n\n[ cat crate ] -> [ player ]\n\n==============\nWINCONDITIONS\n==============\n\nall Target on Crate\n\n=======\nLEVELS\n=======\n\n#########\n#.......#\n#.....@.#\n#.P.*.O.#\n#.......#\n#.......#\n#########\n\n",[2,0,1,1,1,2,2,3,3,3,3,3],"background wall:0,0,0,0,0,0,0,0,background:1,\n1,1,1,1,0,0,1,1,1,\n1,1,0,0,1,1,background player:2,1,1,\n0,0,1,1,background crate:3,1,1,0,0,\n1,1,1,1,1,0,0,1,1,\nbackground target:4,1,2,0,0,1,1,1,1,\n1,0,0,0,0,0,0,0,0,\n",0,"1782027146669.707",[]]
	],
	[
		"a warning raised every turn does not stop the game after 100 turns",
		["title noisy rule group\nauthor test\n\n========\nOBJECTS\n========\n\nBackground\nblack\n\nWall\ngrey\n\nPlayer\nred\n\nLamp\nyellow\n\nDark\nblue\n\n=======\nLEGEND\n=======\n\n. = Background\n# = Wall\nP = Player\nL = Lamp\n\n=======\nSOUNDS\n=======\n\n================\nCOLLISIONLAYERS\n================\n\nBackground\nLamp, Dark\nPlayer, Wall\n\n======\nRULES\n======\n\n[ Lamp ] -> [ Dark ]\n+ [ Dark ] -> [ Lamp ]\n\n==============\nWINCONDITIONS\n==============\n\n=======\nLEVELS\n=======\n\n#######\n#P...L#\n#######\n\n", [3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3,1,3], "background wall:0,0,0,0,background:1,0,0,\nbackground player:2,0,0,1,0,0,1,\n0,0,background lamp:3,0,0,0,0,\n", 0]
	],


	[
		"Selene's Labyrinth",
		[{file:"test_scripts/selene_s_labyrinth.txt"},["tick","tick","tick","tick","tick",0,"tick",0,0,"tick",3,"tick",0,"tick",0,0,"tick","tick"],{file:"test_endstates/selene_s_labyrinth.txt"},0,"1792681669373.4717",[]]
		],
		

		[
			"ENIGMASH",
			[{file:"test_scripts/enigmash.txt"},[1,2,3,2],{file:"test_endstates/enigmash.txt"},1,"1792698495236.9656",[]]
			],

			
[
	"Lock & Pin",
	[{file:"test_scripts/lock_pin.txt"},[0,0,1,0,3,0,1,0,1,2,2,3,0,3,3,3,2,1,3,0,1,1,1,2,3,0,0,3,3,1,1,2,2,1,2,2,2,3,3,3,0,0,2,2,1,1,1,0,1,0,0,3,0,0,3,3,2,1,1,1,1,0,1,1,2,3,0,0,3,2,2],"background wall:0,0,0,0,0,0,0,0,0,\n0,background:1,1,background tunnelb:2,2,background lockb:3,1,1,0,\n0,1,1,2,background lockb pin:4,2,1,1,0,\n0,1,1,1,1,background pin:5,1,1,0,\n0,1,background player tunnela:6,background locka:7,background tunnela:8,1,1,1,0,\n0,1,8,8,7,5,1,1,0,\n0,1,1,1,5,1,1,1,0,\n0,1,1,1,1,1,1,1,0,\n0,0,0,0,0,0,0,0,0,\n",4,"1792795090127.7627",[]]
	],
		
[
	"Being Alan Hazelden",
	[{file:"test_scripts/being_alan_hazelden.txt"},[0,3,3,3,1,1,1,2,2,2,2,3,1,3,0],{file:"test_endstates/being_alan_hazelden.txt"},7,"1792795131045.2227",[]]
	],
	
[
	"Seven-Layer Dip",
	[{file:"test_scripts/seven_layer_dip.txt"},[2,2,2,3,0,0,0,"restart",0,3,2,3,2,1,0,1,1,2,3,3,0,3,2,2,"restart",2,2,3,0,0,1,2,3,2,3,0,1,1,"undo","undo",3,3,0,1,1,0,1,2,2,3,3,2,2,1,0,0,0],{file:"test_endstates/seven_layer_dip.txt"},0,"1792795163315.9407",["36772507","36772507","36772507","36772507","36772507","36772507","36772507","36772507","36772507"]]
	],
	
[
	"Omniban",
	[{file:"test_scripts/omniban.txt"},[3,0,3,0,0,0,0,0,3,3,3,3,3,2,2,2,2,3,3,0,1,2,1,1,1,0,3,3,3,3,2,2,1,1,3,0,3,1,1,1,0,0,1,1,0,0,3,3,3],"background:0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,\n0,0,0,0,0,0,0,0,0,0,0,\n0,background hole:1,1,0,1,1,1,0,1,1,0,\n0,background human:2,0,0,0,0,0,0,background wall:3,3,background omni1 target wall:4,\n3,3,3,0,3,3,3,3,background player target:5,3,3,\n3,0,3,3,3,3,3,3,3,3,0,\n3,3,3,3,3,3,3,3,3,3,3,\n",11,"1792795211394.7292",["2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107","2065107"]]
	],
	
[
	"Un-Lock",
	[{file:"test_scripts/un_lock.txt"},[1,1,1,0,0,0,0,0,0,"undo","undo",3,3,3,3,0,2,1,1,1,0,1,0,0,0,0,0,0,0,3,3,3,0,3,1,0,0,1,0,0,0,0,0,0,1,0,0,3,3,0,2,1,2,2,2,3,3,2,3,3,2,3,3,3,3,3,2,3,1,1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,"undo","undo","undo","undo","undo","undo","undo","undo","undo",1,1,1,0,0,0,0,0,0,3,0,0,0,"undo","undo",1,1,3,0,3,3,0,0,0,0,1,1],{file:"test_endstates/un_lock.txt"},0,"1792795235795.1404",["56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507","56954507"]]
	],[
		"rigid ownership: overlapping rules roll back on action (#1206)",
		["title Overlapping rigid ownership\n\nOBJECTS\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\nMarker\nred\n\nLEGEND\n. = Background\nP = Player and Crate\n\nSOUNDS\n\nCOLLISIONLAYERS\nBackground\nPlayer\nCrate\nWall\nMarker\n\nRULES\nrigid right [ Player ] -> [ > Player Marker ]\nrigid down [ Player Crate ] -> [ > Player > Crate ]\nrigid right [ Wall ] -> [ > Wall ]\n\nWINCONDITIONS\n\nLEVELS\nP\n",[4],"background crate player:0,\n"]
	],
	[
		"rigid ownership: overlapping rules roll back on a no-input tick (#1206)",
		["title Overlapping rigid ownership\n\nOBJECTS\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\nMarker\nred\n\nLEGEND\n. = Background\nP = Player and Crate\n\nSOUNDS\n\nCOLLISIONLAYERS\nBackground\nPlayer\nCrate\nWall\nMarker\n\nRULES\nrigid right [ Player ] -> [ > Player Marker ]\nrigid down [ Player Crate ] -> [ > Player > Crate ]\nrigid right [ Wall ] -> [ > Wall ]\n\nWINCONDITIONS\n\nLEVELS\nP\n",["tick"],"background crate player:0,\n"]
	],
	[
		"rigid ownership: overlapping rules across movement words on action (#1206)",
		["title Overlapping rigid ownership\n\nOBJECTS\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\nMarker\nred\n\nPaddingA\nblack\n\nPaddingB\nblack\n\nPaddingC\nblack\n\nPaddingD\nblack\n\nPaddingE\nblack\n\nLEGEND\n. = Background\nP = Player and Crate\n\nSOUNDS\n\nCOLLISIONLAYERS\nBackground\nPaddingA\nPaddingB\nPaddingC\nPaddingD\nPaddingE\nPlayer\nCrate\nWall\nMarker\n\nRULES\nrigid right [ Player ] -> [ > Player Marker ]\nrigid down [ Player Crate ] -> [ > Player > Crate ]\nrigid right [ Wall ] -> [ > Wall ]\n\nWINCONDITIONS\n\nLEVELS\nP\n",[4],"background crate player:0,\n"]
	],
	[
		"rigid ownership: overlapping rules across movement words on a no-input tick (#1206)",
		["title Overlapping rigid ownership\n\nOBJECTS\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\nMarker\nred\n\nPaddingA\nblack\n\nPaddingB\nblack\n\nPaddingC\nblack\n\nPaddingD\nblack\n\nPaddingE\nblack\n\nLEGEND\n. = Background\nP = Player and Crate\n\nSOUNDS\n\nCOLLISIONLAYERS\nBackground\nPaddingA\nPaddingB\nPaddingC\nPaddingD\nPaddingE\nPlayer\nCrate\nWall\nMarker\n\nRULES\nrigid right [ Player ] -> [ > Player Marker ]\nrigid down [ Player Crate ] -> [ > Player > Crate ]\nrigid right [ Wall ] -> [ > Wall ]\n\nWINCONDITIONS\n\nLEVELS\nP\n",["tick"],"background crate player:0,\n"]
	],
	[
		"rigid ownership: a later rule still owns newly affected layers (#1206)",
		["title Overlapping rigid ownership\n\nOBJECTS\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\nMarker\nred\n\nLEGEND\n. = Background\nP = Player and Crate\nC = Crate\n\nSOUNDS\n\nCOLLISIONLAYERS\nBackground\nPlayer\nCrate\nWall\nMarker\n\nRULES\nrigid right [ Player ] -> [ > Player Marker ]\nrigid down [ Player Crate ] -> [ > Player > Crate ]\nrigid right [ Wall ] -> [ > Wall ]\n\nWINCONDITIONS\n\nLEVELS\nP\nC\n",[4],"background crate player:0,\nbackground crate:1,\n"]
	],
	[
		"rigid ownership: a later rule still owns new layers across movement words (#1206)",
		["title Overlapping rigid ownership\n\nOBJECTS\n\nBackground\nblack\n\nPlayer\nwhite\n\nCrate\norange\n\nWall\ngray\n\nMarker\nred\n\nPaddingA\nblack\n\nPaddingB\nblack\n\nPaddingC\nblack\n\nPaddingD\nblack\n\nPaddingE\nblack\n\nLEGEND\n. = Background\nP = Player and Crate\nC = Crate\n\nSOUNDS\n\nCOLLISIONLAYERS\nBackground\nPaddingA\nPaddingB\nPaddingC\nPaddingD\nPaddingE\nPlayer\nCrate\nWall\nMarker\n\nRULES\nrigid right [ Player ] -> [ > Player Marker ]\nrigid down [ Player Crate ] -> [ > Player > Crate ]\nrigid right [ Wall ] -> [ > Wall ]\n\nWINCONDITIONS\n\nLEVELS\nP\nC\n",[4],"background crate player:0,\nbackground crate:1,\n"]
	],
	[
		"#1207 rejected late randomdir does not leave movement pending after undo",
		// Dust's movement is in the second movement word, beyond restoreLevel's clearing range.
		["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nPaddingA\nblack\n\nPaddingB\nblack\n\nPaddingC\nblack\n\nPaddingD\nblack\n\nPaddingE\nblack\n\nDust D\norange\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\nPaddingA\nPaddingB\nPaddingC\nPaddingD\nPaddingE\nDust\n\nRULES\n\nlate [ Dust ] -> [ randomdir Dust ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n.D.\n...\n", [3, "undo", 4], "background player:0,background:1,1,\n1,background dust:2,1,\n1,1,1,\n", 0, "1207"]
	],
	[
		"#1175 rejected AGAIN must not save a speculative checkpoint",
		["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nTarget X\ngreen\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nTarget\nPlayer\n\nRULES\n\n[ moving Player ] -> [ moving Player ] again\n[ stationary Player Target ] -> [ stationary Player Target ] checkpoint\n\nWINCONDITIONS\nLEVELS\n\nPX\n", [3, "restart"], "background player:0,background target:1,\n"]
	],
	[
		"#1175 a real no-input turn still saves a checkpoint",
		["realtime_interval 1.0\nOBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nTarget X\ngreen\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nTarget\nPlayer\n\nRULES\n\n[ moving Player ] -> [ moving Player ] again\n[ stationary Player Target ] -> [ stationary Player Target ] checkpoint\n\nWINCONDITIONS\nLEVELS\n\nPX\n", [3, "tick", 1, "restart"], "background:0,background player target:1,\n"]
	],
	[
		"#1179 randomdir on the right still moves the player",
		["realtime_interval 0.1\n\nOBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n[ stationary Player ] -> [ randomdir Player ]\n\nWINCONDITIONS\nLEVELS\n\n...\n.P.\n...\n", ["tick"], "background:0,0,0,\nbackground player:1,0,0,\n0,0,0,\n", 0, "1179"]
	],
	[
		"#1179 rejected randomdir drops the whole rule and preserves the following rule",
		["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nCrate C\norange\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer, Crate\n\nRULES\n\nright [ Player | randomdir Crate ] -> [ Player | Crate ]\n[ action Player ] -> [ right Player ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n", [4], "background:0,background player:1,0,\n"]
	],
	[
		"#1179 rejected randomdir drops command-only rules",
		["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer\n\nRULES\n\n[ randomdir Player ] -> restart\n[ action Player ] -> [ right Player ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n", [4], "background:0,background player:1,0,\n"]
	],
	[
		"#1179 rejected random also drops the whole rule",
		["OBJECTS\n\nBackground .\nblack\n\nPlayer P\nwhite\n\nCrate C\norange\n\nLEGEND\nSOUNDS\nCOLLISIONLAYERS\n\nBackground\nPlayer, Crate\n\nRULES\n\nright [ Player | random Crate ] -> [ Player | Crate ]\n[ action Player ] -> [ right Player ]\n\nWINCONDITIONS\nLEVELS\n\nP..\n", [4], "background:0,background player:1,0,\n"]
	],
];

