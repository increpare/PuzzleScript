module PuzzleScript::Syntax

layout LAYOUTLIST = LAYOUT* !>> [\t\ \r(];
lexical LAYOUT
  = [\t\ \r]
  | Comment;

lexical Comment = @lineComment @category="Comment" "(" (![()]|Comment)* ")";
lexical Delimiter = [=]+;
lexical Newline = [\n];
lexical ID = [a-z0-9.A-Z#_+]+ !>> [a-z0-9.A-Z#_+] \ Keywords;
lexical Pixel = [a-zA-Zぁ-㍿.!@#$%&*0-9\-,`\'~_\"§è!çàé;?:/+°£^{}|\>\<^v¬\[\]˅\\±];
lexical LegendKey = @category="LegendKey" Pixel+ !>> Pixel \ Keywords;
lexical SpriteP = [0-9.];
lexical LevelPixel = @category="LevelPixel" Pixel;
lexical Levelline = LevelPixel+ !>> LevelPixel \ Keywords;
lexical String = @category="String" ![\n]+ >> [\n];
lexical SoundIndex = [0-9]|'10' !>> [0-9]|'10';
lexical KeywordID = @category="ID" [a-z0-9.A-Z_]+ !>> [a-z0-9.A-Z_] \ 'message';
lexical IDOrDirectional = @category="IDorDirectional" [\>\<^va-z0-9.A-Z#_+]+ !>> [\>\<^va-z0-9.A-Z#_+] \ Keywords;

keyword SectionKeyword =  'RULES' | 'OBJECTS' | 'LEGEND' | 'COLLISIONLAYERS' | 'SOUNDS' | 'WINCONDITIONS' | 'LEVELS';
keyword PreludeKeyword 
  = 'title' | 'author' | 'homepage' | 'color_palette' | 'again_interval' | 'background_color' 
  | 'debug' | 'flickscreen' | 'key_repeat_interval' | 'noaction' | 'norepeat_action' | 'noundo'
  | 'norestart' | 'realtime_interval' | 'require_player_movement' | 'run_rules_on_level_start' 
  | 'scanline' | 'text_color' | 'throttle_movement' | 'verbose_logging' | 'youtube ' | 'zoomscreen';

keyword LegendKeyword = 'or' | 'and';
keyword CommandKeyword = 'again' | 'cancel' | 'checkpoint' | 'restart' | 'win';
keyword Keywords = SectionKeyword | PreludeKeyword | LegendKeyword | CommandKeyword;

start syntax PSGame
   = @Foldable game: Prelude? Section+
   | game_empty: Newlines;

syntax Newlines = Newline+ !>> [\n];
syntax SectionDelimiter = Delimiter Newline;

syntax Section
   = @Foldable s_objects: SectionDelimiter? 'OBJECTS' Newlines SectionDelimiter? ObjectData+ objects
   | @Foldable s_legend: SectionDelimiter? 'LEGEND' Newlines SectionDelimiter? LegendData+ legend
   | @Foldable s_sounds: SectionDelimiter? 'SOUNDS' Newlines SectionDelimiter? SoundData+ sounds
   | @Foldable s_layers: SectionDelimiter? 'COLLISIONLAYERS' Newlines SectionDelimiter? LayerData+ layers
   | @Foldable s_rules: SectionDelimiter? 'RULES' Newlines SectionDelimiter? RuleData+ rules
   | @Foldable s_conditions: SectionDelimiter? 'WINCONDITIONS' Newlines SectionDelimiter? ConditionData+ conditions
   | @Foldable s_levels: SectionDelimiter? 'LEVELS' Newlines SectionDelimiter? LevelData+ levels
   | s_empty: SectionDelimiter? SectionKeyword Newlines SectionDelimiter?
   ;
 	
syntax Prelude
  = prelude: PreludeData+
  ;
	
syntax PreludeData
  = prelude_data: KeywordID key String* string Newline
  | prelude_empty: Newline;
  
syntax Sound
  = sound: 'sfx' SoundIndex;	
		
syntax Color
  = @category="Color" ID;
	
syntax ObjectName
  = @category="ObjectName" ID;

syntax ObjectData
  = @Foldable object_data: ObjectName LegendKey? Newline Colors Newline Sprite?
  | object_empty: Newline;
	
syntax SpritePixel
  = @category="SpritePixel" SpriteP;

syntax Colors
  = Color+;

syntax Sprite 
  = @Foldable sprite: 
    SpritePixel+ Newline
    SpritePixel+ Newline
    SpritePixel+ Newline 
    SpritePixel+ Newline
    SpritePixel+ Newline;
	
syntax LegendOperation
  = legend_or: 'or' ObjectName
  | legend_and: 'and' ObjectName;

syntax LegendData
  = legend_data: LegendKey '=' ObjectName LegendOperation*  Newline
  | legend_empty: Newline;
	
syntax SoundID
  = @category="SoundID" ID;
	
syntax SoundData
  = sound_data: SoundID+ Newline
  | sound_empty: Newline;

syntax LayerData
  = layer_data: (ObjectName ','?)+ Newline
  | layer_empty: Newline;
	
syntax RuleData
  = rule_data: (Prefix|RulePart)+ '-\>' (Command|RulePart)* Message? Newline
  | @Foldable rule_loop: 'startloop' RuleData+ 'endloop' Newline
  | rule_empty: Newline;

syntax RuleContent
  = content: IDOrDirectional*;
	
syntax RulePart
  = part: '[' {RuleContent '|'}+ ']';

syntax Prefix
  = @category="Keyword" prefix: ID;

syntax Command
  = @category="Keyword" command: CommandKeyword
  | @category="Keyword" sound: Sound;
	
syntax ConditionID
  = @category="ConditonID" ID;

syntax ConditionData
  = condition_data: ConditionID+ Newline
  | condition_empty: Newline;
  
syntax Message
  = 'message' String*;

syntax LevelData
  = @Foldable level_data_raw: (Levelline Newline)+ lines Newline
  | message: 'message' String*
  | level_empty: Newline;