#/usr/bin/env python3

# MIT License
# Copyright (c) 2020 Jens Weggemann
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import re
import argparse  
import math
from collections import namedtuple
from enum import Enum, auto
from types import SimpleNamespace
from pathlib import Path

parser = argparse.ArgumentParser(description='generates annotated HOVALAAG traces.')
parser.add_argument('-c', '--code', metavar='FILE', required=True, help='path to the code input file, required')
parser.add_argument('-l', '--log', metavar='FILE', default='log.html', help='path to the trace input file, defaults to "log.html"')
parser.add_argument('-o', '--output', metavar='FILE', help='path to the HTML output file, stdout is used if omitted')
parser.add_argument('-f', '--force', action='store_true', help='overwrite the output file if it exists')
parser.add_argument('-n', '--numbers', choices=['s', 'u', 'h', 'b'], default='s', help='display type for numbers: [s]igned decimal, [u]nsigned decimal, [h]exadecimal, or [b]inary, defaults to s')
parser.add_argument('--strip-comments', action='store_true', help='strip all code comments')
parser.add_argument('--comments-across-empty-lines', action='store_true', help='if specified, empty lines will not disassociate comment lines with the statement succeeding them')
parser.add_argument('--always-print-comment-lines', action='store_true', help='if specified, comment lines will be printed for a statement every time, by default they are only printed the first time')
parser.add_argument('--theme', choices=['dark', 'light'], help='the color theme, defaults to dark')
args = parser.parse_args()

def err(*argv):
    print(*argv, file=sys.stderr)
    sys.exit(-1)

if not args.code:
    err("Must specify a code file!")

#====================================================================
# Helpers

def dec_to_2scompl(num, num_bits):
    n = min(int(num), (1 << (num_bits - 1)) - 1)
    if n < 0: n = (1 << num_bits) + n
    return n

def dec_from_2scompl(num, num_bits):
    n = int(num) & ((1 << num_bits) - 1)
    ceiling = (1 << num_bits)
    if n >= (1 << (num_bits - 1)): n -= (1 << num_bits)
    return n

def as_hex(n, num_bits):
    s = '{:X}'.format(n)
    return '$' + '0' * max(0, num_bits // 4 - len(s)) + s

def as_binary(n, num_bits):
    s = '{:b}'.format(n)
    return 'b' + '0' * max(0, num_bits - len(s)) + s

def make_number_text(n, num_bits):
    if args.numbers == 's': return n
    n_2sc = dec_to_2scompl(n, num_bits)
    if args.numbers == 'u': return n_2sc + 'u'
    elif args.numbers == 'h': return as_hex(n_2sc, num_bits)
    elif args.numbers == 'b': return as_binary(n_2sc, num_bits)

def make_number_tooltip(n, num_bits):
    n_2sc = dec_to_2scompl(n, num_bits)
    return "{} {}u {} {}".format(n, n_2sc, as_hex(n_2sc, num_bits).upper(), as_binary(n_2sc, num_bits))

#====================================================================
# Parse code

CodeLine = namedtuple('CodeLine', ['line_num', 'text'])
# line: CodeLine, comments: CodeLine list, label: CodeLine
Statement = namedtuple('Statement', ['pc_line_num', 'line', 'comments', 'labels'])
statements_by_pc = {}

re_line_strip_comments = re.compile(r'^(.*?);.*$')
re_line_remove_trailing_ws = re.compile(r'^(.*?)[\s\t\n\r]*$')
re_line_empty = re.compile(r'^\s*$')
re_code_comment_line = re.compile(r'^\s*;.*$')
re_code_label = re.compile(r'^\s*([^\s]+):.*$')
curr_comments = []
curr_labels = []
file_line_num = 0
pc_line_num = 0
with open(args.code) as f:
    for line in f.readlines():
        file_line_num += 1

        # Optionally strip any comments. Line may become empty.
        if args.strip_comments:
            match = re_line_strip_comments.match(line)
            if match: line = match.group(1)

        # Remove any trailing WS
        match = re_line_remove_trailing_ws.match(line)
        line = match.group(1)

        # Handle empty line
        match = re_line_empty.match(line)
        if match: 
            if not args.comments_across_empty_lines:
                # Forget comments but keep a label
                curr_comments = []
            continue

        code_line = CodeLine(file_line_num, line)
        match = re_code_comment_line.match(line)
        if match:
            curr_comments.append(code_line)
            continue
        match = re_code_label.match(line)
        if match:
            curr_labels.append(code_line)
            continue

        # Everything else must be a statement.
        # We commit it to the result and reset relevant parsing state.
        statements_by_pc[pc_line_num] = Statement(pc_line_num, code_line, curr_comments, curr_labels)
        curr_labels = []
        curr_comments = []
        pc_line_num += 1

file_line_num_digits = math.ceil(math.log(file_line_num, 10))

#====================================================================
# Parse the trace log

# We'll treat the expected value of O1 and O2 (if any) as pseudo-registers for convenience.
REGISTERS = ['ALU', 'A', 'B', 'C', 'D', 'F', 'W', 'I1', 'I2', 'O1', 'O1e', 'O2', 'O2e', 'PC']

TraceLine = namedtuple('TraceLine', REGISTERS, defaults=[0]*len(REGISTERS))
ChangedRegisters = namedtuple('ChangedRegisters', REGISTERS, defaults=[False]*len(REGISTERS))
trace_lines = []
changed_registers = []

def parse_trace_number(n, R):
    # Parses a number constant from the trace. Interpretation depends on the specific register.
    if n is None: return None
    n = int(n, 16) # Parse hex
    if R == 'PC': # PC is unsigned
        pass
    elif R == 'ALU': # ALU is 16-bit 2s-complement signed
        n = dec_from_2scompl(n, 16)
    else: # Everything else is 12-bit 2s-complement signed

        # The hovalaag trace shows W as if it was a 64-bit value when it has a negative value.
        # This is contrary to the documentation. Copying to OUT1 drops the upper bits again.
        # Is this a bug? Anyway, the helper function will mask out all but the lower 12 bits which
        # fixed this.
        n = dec_from_2scompl(n, 12)
    return n

REGISTERS = ['ALU', 'A', 'B', 'C', 'D', 'F', 'W', 'I1', 'I2', 'O1', 'O1e', 'O2', 'O2e', 'PC']

re_log_line = re.compile(r'<tr.*?</tr>')
re_log_line_entry = re.compile(r'(ALU|I1|I2|O1|O2|PC|[ABCDFW])[=:]([0-9a-fA-F]+)(?:\(([0-9a-fA-F]+)\))?')

prev_trace_line = TraceLine()
with open(args.log) as f:
    counter = -1
    for line in f.readlines():
        counter += 1
        match = re.match(re_log_line, line)
        # Collect in dict first b/c the TraceLine namedtupe is immutable. Not very elegant.
        values = {k:None for k in REGISTERS}
        if match:
            line = match.group(0)
            for match in re.finditer(re_log_line_entry, line):
                R, reg_value, expected = match.groups()[:3]
                values[R] = parse_trace_number(reg_value, R)
                if R == 'O1': values['O1e'] = parse_trace_number(expected, R)
                elif R == 'O2': values['O2e'] = parse_trace_number(expected, R)
                elif R == 'PC': break # terminate before the instruction parts.

            trace_line = TraceLine(*[values[R] for R in REGISTERS])
            trace_lines.append(trace_line)
            changed_registers.append(ChangedRegisters(*[values[R] != getattr(prev_trace_line, R) for R in REGISTERS]))
            prev_trace_line = trace_line

#====================================================================
# Syntax highlighting machinery

class TokenTag(Enum):
    COMMENT = auto()
    LABEL = auto()
    CONSTANT = auto()
    JUMP = auto()
    OPERATOR = auto()
    FUNCTION = auto()
    DSTREGISTER = auto()
    SRCREGISTER = auto()
    WHITESPACE = auto()
    UNKNOWN = auto()

class Tokenizer(object):
    def __init__(self):
        self.rules = [
            (TokenTag.COMMENT, re.compile(r';.*$')),
            (TokenTag.LABEL, re.compile(r'[^\s]+:')),
            (TokenTag.DSTREGISTER, re.compile(r'([ABCDFW]|IN[12]|OUT[12])(?==)')),
            (TokenTag.JUMP, re.compile(r'(JMP[TF]?|DECNZ)\s+([^\s,]+)')),
            (TokenTag.FUNCTION, re.compile(r'(ZERO|NEG|POS|DEC)|[()]')),
            (TokenTag.SRCREGISTER, re.compile(r'([ABCDFW]|IN[12]|OUT[12])')),
            (TokenTag.CONSTANT, re.compile(r'-?(([0-9]+)|(\$[0-9A-F]+))')),
            (TokenTag.OPERATOR, re.compile(r'[+->|&^~]')),
            (TokenTag.WHITESPACE, re.compile(r'\s+')),
            (TokenTag.UNKNOWN, re.compile(r'.')),
        ]

    def tokenize_line(self, text):
        tokens = []
        last_token = None
        pos = 0
        while pos < len(text):
            for tag, regexp in self.rules:
                match = regexp.match(text, pos=pos)
                if match:
                    # Some token types are collapsed right away.
                    if last_token and last_token[0] == tag and tag in [TokenTag.WHITESPACE, TokenTag.UNKNOWN]:
                        last_token[1] += match.group(0)
                    else:
                        token = [tag, match.group(0)]
                        tokens.append(token)
                        last_token = token
                    pos = match.end(0)
                    break
        return tokens

#====================================================================
# Output HTML

ColorTheme = namedtuple("ColorTheme", [
    'bg',
    'bg_regstate',
    'bg_regstate_handle',
    'bg_changedreg',
    'jump_separator',
    'wrong_output_border',
    'correct_output_border',
    'syn_dstregister',
    'syn_srcregister',
    'syn_operator',
    'syn_outputline',
    'syn_linenumber',
    'syn_comment',
    'syn_label',
    'syn_jump',
    'syn_function',
    'syn_constant',
    'line_marker1',
    'line_marker2',
    'line_marker3',
    'bg_tooltip',
    'fg_tooltip',
    'tooltip_elem_bg' ])

# Based on light style from https://github.com/rakr/vim-two-firewatch 
LightTheme = ColorTheme(
    bg='#FAF8F5',
    bg_regstate='#eAe8e5',
    bg_regstate_handle='#0003',
    bg_changedreg='#ffffff',
    jump_separator='#2D2107',
    wrong_output_border='#ff9000',
    correct_output_border='#20f030',
    syn_dstregister='#718ECD',
    syn_srcregister='#718ECD',
    syn_operator='#896724',
    syn_outputline='#E4DBD7',
    syn_linenumber='#c2aEa7',
    syn_comment='#B6AD9A',
    syn_label='#2D2107',
    syn_jump='#0A5289',
    syn_function='#896724',
    syn_constant='#0A5289',
    line_marker1='#FDF962',
    line_marker2='#B6FB8B',
    line_marker3='#FCC8B8',
    bg_tooltip='#0A5289d0',
    fg_tooltip='#FAF8F5',
    tooltip_elem_bg='#ffe04850')

# Based on dark style from https://github.com/rakr/vim-two-firewatch 
DarkTheme = ColorTheme(
    bg='#282c34',
    bg_regstate='#11151D',
    bg_regstate_handle='#000000',
    bg_changedreg='#32363E',
    jump_separator='#8E9DAE',
    wrong_output_border='#e02000',
    correct_output_border='#20d010',
    syn_dstregister='#D6E9FF',
    syn_srcregister='#D6E9FF',
    syn_operator='#8EBCF2',
    syn_outputline='#3D4854',
    syn_linenumber='#616C78',
    syn_comment='#55606C',
    syn_label='#D5E8FD',
    syn_jump='#DE6A6F',
    syn_function='#C4AB9A',
    syn_constant='#eaAE9D',
    line_marker1='#181C24',
    line_marker2='#0F4909',
    line_marker3='#702714',
    bg_tooltip='#0A5289ff',
    fg_tooltip='#FAF8F5',
    tooltip_elem_bg='#ff803050')

if args.theme == 'light':
    color_theme = LightTheme
else:
    color_theme = DarkTheme

if args.output and Path(args.output).is_file() and not args.force:
    err("Output file exists, use --force to overwrite!")

output_file = open(args.output, "w") if args.output else sys.stdout
def pr(*argv, **kwargv):
    print(*argv, **kwargv, file=output_file)

pr('''\
<html>
<head>
<style>
body {{
    font-family: monospace;
    background-color: {bg};
    position: relative;
    left: 0;
    top: 0;
    padding-top: 2ch;
}}
#code {{
    white-space: pre;
}}
#regstate {{
    white-space: pre;
    background-color: {bg_regstate};
    opacity: .9;
    position: absolute;
    left: 50ch;
    top: 2ch;
    padding-left: 3px;
}}
#regstate_handle {{
    width: 1ch; 
    height: 100%;
    text-align: center;
    left: 0;
    top: 0;
    cursor: col-resize;
    background: {bg_regstate_handle};
    position: absolute;
}}
.reg_unchanged {{
    opacity: 0.7;
}}
.reg_changed {{
    background-color: {bg_changedreg};
}}
.reg_value_changed {{
}}
.reg_value_unchanged {{
}}

.wrong_output_value {{
    border-width: 1px 3px;
    border-style: solid;
    border-color: {wrong_output_border};
    display: inline-block;
    z-index: 100;
}}
.correct_output_value {{
    border-width: 1px 3px;
    border-style: solid;
    border-color: {correct_output_border};
    display: inline-block;
    z-index: 100;
}}

.code_line {{
}}

.linebg0 {{
}}
.linebg1 {{
    background-color: {line_marker1};
}}
.linebg2 {{
    background-color: {line_marker2};
}}
.linebg3 {{
    background-color: {line_marker3};
}}

.syn_outputline {{
    color: {syn_outputline};
    cursor: hand;
}}
.syn_linenumber {{
    color: {syn_linenumber};
    cursor: hand;
}}
.syn_comment {{
    color: {syn_comment};
    font-style: italic;
}}
.syn_label {{
    color: {syn_label};
    font-weight: bold;
}}
.syn_dstregister {{
    color: {syn_dstregister};
}}
.syn_srcregister {{
    color: {syn_srcregister};
}}
.syn_jump {{
    color: {syn_jump};
}}
.syn_function {{
    color: {syn_function};
}}
.syn_constant {{
    color: {syn_constant};
}}
.syn_operator {{
    color: {syn_operator};
}}
.syn_jump_not_taken {{
    text-decoration: line-through;
}}

.jump_separator {{
  border-top: 1px solid;
  border-color: {jump_separator};
  background-color: transparent;
  opacity: 0.5;
  width: 100%;
  position: absolute;
  z-index: 5;
}}

/* Tooltip CSS from https://www.w3schools.com/css/css_tooltip.asp */
.tooltip {{
  position: relative;
  display: inline-block;
  z-index: 100;
}}
.tooltip:hover {{
    background-color: {tooltip_elem_bg};
}}
.tooltip .tooltiptext {{
  visibility: hidden;
  pointer-events: none; /* invisible to hover */
  background-color: {bg_tooltip};
  color: {fg_tooltip};
  text-align: center;
  border-radius: 3px;
  padding: 5px 5px;
  bottom: 110%;
  left: -100%;
  position: absolute;
  z-index: 100;
  filter: drop-shadow(4px 3px 2px #00000080);
}}
.tooltip:hover .tooltiptext {{
  visibility: visible;
}}

</style>
<script>

function line_click(the_id) {{
    let elem = document.getElementById(the_id);
    let cl = elem.classList;
    const CYCLE = ["linebg0", "linebg1", "linebg2", "linebg3"];
    for (let i = 0; i < CYCLE.length; ++i) {{
        const next_i = (i + 1) % CYCLE.length;
        if (cl.contains(CYCLE[i])) {{
            cl.remove(CYCLE[i]);
            cl.add(CYCLE[next_i]);
            return;
        }}
    }}
    cl.add(CYCLE[1]);
}}

</script>
</head>
<body><div id="code">\
'''.format(**color_theme._asdict()), end='')

#====================================================================
# Output code

def css_for_tag(tag):
    return tag.name.lower()

def padded_line_num(num, digits):
    s = str(num)
    return (' ' * max(0, digits - len(s))) + s

tokenizer = Tokenizer() 
output_line_digits = 4 # Currently a guess.. we'd have to collect output lines beforehand and prepend afterwards.
output_line_counter = 0

def is_jump_taken(token, trace_line):
    if re.match(r'^JMPT\s+.*', token):
        return trace_line.F != 0
    elif re.match(r'^JMPF\s+.*', token):
        return trace_line.F == 0
    elif re.match(r'^DECNZ\s+.*', token):
        return trace_line.C != 0
    return True

def output_line(line, trace_line=None):
    global output_line_counter
    output_line_counter += 1
    pr('<span class="code_line" id="line{}" onclick="line_click(id)">'.format(output_line_counter), end='')
    pr('<span class="syn_outputline">{}</span> '.format(padded_line_num(output_line_counter, output_line_digits)), end='')
    pr('<span class="syn_linenumber">{}</span> '.format(padded_line_num(line.line_num, file_line_num_digits)), end='')
    tokens = tokenizer.tokenize_line(line.text)
    for tok_tag, tok_value in tokens:
        if tok_tag is TokenTag.CONSTANT:
            # Convert constant text and add tooltip
            if tok_value.startswith('$'): num = int(tok_value[1:], 16)
            elif tok_value.startswith('-$'): num = int(tok_value[2:], 16)
            else: num = int(tok_value)
            num_2sc = dec_to_2scompl(num, 12)
            pr('<div class="tooltip syn_{}">{}<span class="tooltiptext">{}</span></div>'.format(
                css_for_tag(tok_tag), make_number_text(num, 12), make_number_tooltip(num, 12)), end='')
        else:
            extra_class = ''
            if tok_tag is TokenTag.JUMP and trace_line:
                # We can be extra smart about this!
                if not is_jump_taken(tok_value, trace_line):
                    extra_class = " syn_jump_not_taken"
            pr('<span class="syn_{}{}">{}</span>'.format(css_for_tag(tok_tag), extra_class, tok_value), end='')

    pr('</span>')


PRINT_COMMENT_BLOCKS_ONLY_ONCE = True
comment_blocks_already_printed_for_pc = set()
displayed_trace_info = []

previous_pc = None
for trace_line, changed_regs in zip(trace_lines, changed_registers):
    pc = trace_line.PC
    statement = statements_by_pc[pc]

    if previous_pc and pc != previous_pc + 1:
        # Deviating program counter -> some jump occured.
        pr('<div class="jump_separator"></div>', end='')

    if not args.always_print_comment_lines and pc in comment_blocks_already_printed_for_pc:
        # Don't show comment again, but do show associated labels
        for label in statement.labels:
            output_line(label)
            displayed_trace_info.append(None)
    else:
        comment_blocks_already_printed_for_pc.add(pc)
        if not statement.comments:
            for label in statement.labels:
                output_line(label)
                displayed_trace_info.append(None)
        else:
            # Display associated comments with label mixed in
            for line in sorted(statement.labels + statement.comments, key=lambda elem: elem.line_num):
                output_line(line)
                displayed_trace_info.append(None)

    output_line(statement.line, trace_line)
    displayed_trace_info.append((trace_line, changed_regs))
    previous_pc = pc

#====================================================================
# Output register state

pr('\n\n<div id="regstate"><div id="regstate_handle"></div>', end='')

def properly_padded(n, R):
    max_chars = 6
    if args.numbers == 's': max_chars = 6 if R == 'ALU' else 1 if R == 'F' else 5
    elif args.numbers == 'u': max_chars = 7 if R == 'ALU' else 2 if R == 'F' else 6
    elif args.numbers == 'h': max_chars = 4 if R == 'ALU' else 1 if R == 'F' else 3
    elif args.numbers == 'b': max_chars = 16 if R == 'ALU' else 1 if R == 'F' else 12
    s = str(n)
    # Roughly center the value.. I liked that look the most.
    padding = max_chars - len(s)
    half_padding = padding // 2
    return ' ' * half_padding + s + ' ' * (padding - half_padding)

for thing in displayed_trace_info:
    if not thing: 
        pr()
        continue
    trace_line, changed_regs = thing

    printed_line = ' '
    for R in ['A', 'B', 'C', 'D', 'F', 'W', 'ALU', 'I1', 'I2', 'O1', 'O2']:
        reg_value = getattr(trace_line, R)
        reg_changed = getattr(changed_regs, R)
        if reg_value is None: continue

        overall_css_class = 'reg_{}changed'.format('un' if not reg_changed else '')
        value_css_class = 'reg_value_{}changed'.format('un' if not reg_changed else '')
        reg_bits = 16 if R == 'ALU' else 1 if R == 'F' else 12

        exp_value = None
        if R in ['O1', 'O2']:
            exp_value = getattr(trace_line, 'O1e' if R == 'O1' else 'O2e')

        if exp_value is None or exp_value == reg_value:

            printed_line += '''\
<div class="tooltip {occ}{output_value_style}">\
<span class="syn_dstregister ">{R}</span>\
<span class="syn_operator ">=</span>\
<span class="syn_constant {vcc}">{rv}</span><span class="tooltiptext">{tt}</span></div> '''.format(
                occ=overall_css_class,
                output_value_style=' correct_output_value' if R in ['O1', 'O2'] else '',
                vcc=value_css_class,
                R=R,
                rv=properly_padded(make_number_text(reg_value, reg_bits), R),
                tt=make_number_tooltip(reg_value, reg_bits))

        else:

            printed_line += '''\
<div class="wrong_output_value">\
<div class="tooltip {occ}">\
<span class="syn_dstregister ">{R}</span>\
<span class="syn_operator ">=</span>\
<span class="syn_constant {vcc}">{rv1}</span><span class="tooltiptext">{tt1}</span>\
</div>\
<div class="tooltip {occ}">\
<span class="syn_constant {vcc}">({rv2})</span><span class="tooltiptext">{tt2}</span>\
</div>\
</div> '''.format(
                occ=overall_css_class,
                vcc=value_css_class,
                R=R,
                rv1=properly_padded(make_number_text(reg_value, reg_bits), R),
                tt1=make_number_tooltip(reg_value, reg_bits),
                rv2=make_number_text(exp_value, reg_bits),
                tt2=make_number_tooltip(exp_value, reg_bits))

    pr(printed_line)

pr('''\
</div>
</div>
<script>
// From https://www.w3schools.com/howto/howto_js_draggable.asp

dragElement(document.getElementById("regstate"));

function dragElement(elmnt) {
var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
if (document.getElementById(elmnt.id + "_handle")) {
    /* if present, the header is where you move the DIV from:*/
    document.getElementById(elmnt.id + "_handle").onmousedown = dragMouseDown;
} else {
    /* otherwise, move the DIV from anywhere inside the DIV:*/
    elmnt.onmousedown = dragMouseDown;
}

function dragMouseDown(e) {
    e = e || window.event;
    e.preventDefault();
    // get the mouse cursor position at startup:
    pos3 = e.clientX;
    pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    // call a function whenever the cursor moves:
    document.onmousemove = elementDrag;
}

function elementDrag(e) {
    e = e || window.event;
    e.preventDefault();
    // calculate the new cursor position:
    pos1 = pos3 - e.clientX;
    pos2 = pos4 - e.clientY;
    pos3 = e.clientX;
    pos4 = e.clientY;
    // set the element's new position:
    //elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
    elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
}

function closeDragElement() {
    /* stop moving when mouse button is released:*/
    document.onmouseup = null;
    document.onmousemove = null;
}
}
</script>
</body>
</html>
''')
