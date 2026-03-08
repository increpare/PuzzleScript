class RuleTransform {


    /*******************/

        /* Class for rule transformation (for smart autocomplete)
        e.g. if I type
            up [ > Player | Crate_up ] -> [ > Player | > Crate_up ] 
            d

        the autocomplete should suggest
            down [ > Player | Crate_down ] -> [ > Player | > Crate_down ]

        this should mirror the rule vertically (including any movement directions mentioned)

        if instead i start the next line with 'r', the suggestion should rotate the rule 90 degrees clockwise
            right [ > Player | Crate_right ] -> [ > Player | > Crate_right ]

        stuff like this!
    */

    /*
        So, what does it take to trigger a rule.  Well we need one rule that has a direction prefix, and 
        then we need to start writing the next rule, with a compatible direction prefix.
        so 'up' can be followed by 'down' (mirrors), or left/right (rotates clockwise/counter-clockwise), 
        but not 'up' again or 'horizontal'/'vertical'.
    */

    //the directions that can trigger rule-autocomplete (not strictly needed to hard-code, but good for QOL)
    static RULE_DIRECTION_TOKENS = ['up', 'down', 'left', 'right', 'horizontal', 'vertical'];

    
    
    //the transformations that can be applied to each direction
    static DIR_TRANSFORMATION_RULES = {
        'up': {
            'down': "MIRROR_V",
            'left': "ROTATE_CCW",
            'right': "ROTATE_CW",
        },
        'down': {
            'up': "MIRROR_V",
            'left': "ROTATE_CW",
            'right': "ROTATE_CCW",
        },
        'left': {
            'right': "MIRROR_H",
            'up': "ROTATE_CCW",
            'down': "ROTATE_CW",
        },
        'right': {
            'left': "MIRROR_H",
            'up': "ROTATE_CW",
            'down': "ROTATE_CCW",
        },
        'horizontal': {
            'vertical': "ROTATE_CW",
        },
        'vertical': {
            'horizontal': "ROTATE_CW",
        },
    }

    static directions_compatible(fromDir, toDir) {
        if (!(fromDir in RuleTransform.DIR_TRANSFORMATION_RULES)) 
            return false;
        var transformations = RuleTransform.DIR_TRANSFORMATION_RULES[fromDir];
        if (!(toDir in transformations)) 
            return false;
        return true;        
    }

    static DIRECTIONAL_PAIRINGS = [
        ['Up', ['Down', 'Left', 'Right']],
        ['UP', ['DOWN', 'LEFT', 'RIGHT']],
        ['up', ['down', 'left', 'right']],
        ['_u', ['_d', '_l', '_r']],
        ['_U', ['_D', '_L', '_R']],
        ['North', ['South', 'West', 'East']],
        ['NORTH', ['SOUTH', 'WEST', 'EAST']],
        ['north', ['south', 'west', 'east']],
        ['_n', ['_s', '_w', '_e']],
        ['_N', ['_S', '_W', '_E']],
    ];

    static FLATTENED_DIRECTIONAL_PAIRINGS = [
        ['Up', 'Down', 'Left', 'Right'],
        ['UP', 'DOWN', 'LEFT', 'RIGHT'],
        ['up', 'down', 'left', 'right'],
        ['_u', '_d', '_l', '_r'],
        ['_U', '_D', '_L', '_R'],
        ['North', 'South', 'West', 'East'],
        ['NORTH', 'SOUTH', 'WEST', 'EAST'],
        ['north', 'south', 'west', 'east'],
        ['_n', '_s', '_w', '_e'],
        ['_N', '_S', '_W', '_E'],
    ];


    /** Returns direction token if word is an exact or prefix match (e.g. "d" -> "down"), else null. */
    static matchRuleDirection(word) {
        if (!word) return null;
        var w = (word + '').toLowerCase();
        for (var i = 0; i < RuleTransform.RULE_DIRECTION_TOKENS.length; i++) {
            var d = RuleTransform.RULE_DIRECTION_TOKENS[i];
            if (d === w || d.lastIndexOf(w, 0) === 0) return d;
        }
        return null;
    }

    static getFirstRuleDirection(line) {
        var s = line.trim();
        var commentIdx = s.indexOf('#');
        if (commentIdx >= 0) s = s.substring(0, commentIdx).trim();
        var tokens = s.split(/\s+/);
        for (var i = 0; i < tokens.length; i++) {
            var t = tokens[i].toLowerCase();
            if (RuleTransform.RULE_DIRECTION_TOKENS.indexOf(t) >= 0) return t;
            if (t !== 'late' && t !== 'rigid' && t !== '+' && t !== '') return null;
        }
        return null;
    }

    static stripRuleComment(line) {
        return line.replace(/\s*\([^)]*\)/g, '').replace(/\s+/g, ' ').trim();
    }

    static invariant_tokens = [
        "[", "]", "|", "->", "+", 
        "late", "rigid", 
        "^", "v", "<", ">", 
        "moving", "stationary", "randomdir", "random", "orthogonal", "action", "...", "parallel", "perpendicular", "no"
    ];


    static get_object_directional_suffix(token) {
        for (var i = 0; i < RuleTransform.DIRECTIONAL_PAIRINGS.length; i++) {
            var pairing = RuleTransform.DIRECTIONAL_PAIRINGS[i];
            var suffixes = [pairing[0]].concat(pairing[1]);
            for (var j = 0; j < suffixes.length; j++) {
                var suf = suffixes[j];
                if (token.endsWith(suf)) return suf;
            }
        }
        return null;
    }

    static has_object_directional_suffix(token) {
        return RuleTransform.get_object_directional_suffix(token) !== null;
    }

    static get_object_name_stem(token,suffix,strip_underscores=true) {
        var cleaned = token.slice(0, -suffix.length);
        //trim any trailing underscores
        if (strip_underscores) {
            while (cleaned.endsWith('_')) {
                cleaned = cleaned.slice(0, -1);
            }
        }
        return cleaned;
    }

    static direction_transformations = {
        "up": {
            "ROTATE_CW": "right",
            "ROTATE_CCW": "left",
            "MIRROR_H": "up",
            "MIRROR_V": "down",
        },
        "down": {
            "ROTATE_CW": "left",
            "ROTATE_CCW": "right",
            "MIRROR_H": "down",
            "MIRROR_V": "up",
        },
        "left": {
            "ROTATE_CW": "up",
            "ROTATE_CCW": "down",
            "MIRROR_H": "right",
            "MIRROR_V": "left",
        },
        "right": {
            "ROTATE_CW": "down",
            "ROTATE_CCW": "up",
            "MIRROR_H": "left",
            "MIRROR_V": "right",
        },
        "horizontal": {
            "ROTATE_CW": "vertical",
            "ROTATE_CCW": "vertical",
            "MIRROR_H": "horizontal",
            "MIRROR_V": "horizontal",
        },
        "vertical": {
            "ROTATE_CW": "horizontal",
            "ROTATE_CCW": "horizontal",
            "MIRROR_H": "vertical",
            "MIRROR_V": "vertical",
        },
    };

    static get_direction_from_suffix(suffix) {
        //use FLATTENED_DIRECTIONAL_PAIRINGS for this
        for (var i = 0; i < RuleTransform.FLATTENED_DIRECTIONAL_PAIRINGS.length; i++) {
            var pairings = RuleTransform.FLATTENED_DIRECTIONAL_PAIRINGS[i];
            for (var j = 0; j < pairings.length; j++) {
                var suf = pairings[j];
                if (suffix===suf) return j;
            }
        }
        return null;
    }

    static get_suffix_type_idx_from_suffix(suffix) {
        for (var i = 0; i < RuleTransform.FLATTENED_DIRECTIONAL_PAIRINGS.length; i++) {
            var pairings = RuleTransform.FLATTENED_DIRECTIONAL_PAIRINGS[i];
            for (var j = 0; j < pairings.length; j++) {
                var suf = pairings[j];
                if (suffix===suf) return i;
            }
        }
    }

    static apply_transformation_to_direction(dir, transformation) {
        //dir 0123 = up, down, left, right
        switch (transformation) {
            case "ROTATE_CW":
                return [3,2,0,1][dir];
            case "ROTATE_CCW":
                return [2,3,1,0][dir];
            case "MIRROR_H":
                return [0,1,3,2][dir];
            case "MIRROR_V":
                return [1,0,2,3][dir];
        }
        //should never happen, print error
        console.error("Invalid transformation: " + transformation);
        return -1;
    }

    static transformObject(object_name, transformation) {
        var suffix = RuleTransform.get_object_directional_suffix(object_name);
        var stem = RuleTransform.get_object_name_stem(object_name,suffix,false);
        var dir = RuleTransform.get_direction_from_suffix(suffix);
        var new_dir = RuleTransform.apply_transformation_to_direction(dir, transformation);
        var suffix_type_idx = RuleTransform.get_suffix_type_idx_from_suffix(suffix);
        var new_suffix = RuleTransform.FLATTENED_DIRECTIONAL_PAIRINGS[suffix_type_idx][new_dir];
        return stem + new_suffix;
    }

    static applyTransformationToToken(token, transformation) {
        if (RuleTransform.invariant_tokens.indexOf(token) >= 0) return token;
        if (RuleTransform.direction_transformations[token]) {
            return RuleTransform.direction_transformations[token][transformation];
        }

        //am i an object with a directional suffix?
        if (RuleTransform.has_object_directional_suffix(token)) {
            return RuleTransform.transformObject(token, transformation);
        }
        //now to look for objects with suffixes
        return token;
    }
    static applyTransformation(line, transformation) {
        line = RuleTransform.stripRuleComment(line);
        var tokens = tokenizeRuleLine(line);
        if (!tokens) return null;
        for (var i = 0; i < tokens.length; i++) {
            tokens[i] = RuleTransform.applyTransformationToToken(tokens[i], transformation);            
        }
        return tokens.join(' ');
    }

    static rotateRuleLineToDirection(line, newDir) {
        var fromDir = RuleTransform.getFirstRuleDirection(line);
        if (!fromDir) return null;
        var transformation = RuleTransform.DIR_TRANSFORMATION_RULES[fromDir][newDir];
        var transformed_rule = RuleTransform.applyTransformation(line, transformation);
        return transformed_rule;
    }
    
    static tryTransformRuleLine(line, newDir) {
        var fromDir = RuleTransform.getFirstRuleDirection(line);
        if (!fromDir) return null;
        if (!RuleTransform.directions_compatible(fromDir, newDir)) return null;
        return RuleTransform.rotateRuleLineToDirection(line, newDir);
    }
}