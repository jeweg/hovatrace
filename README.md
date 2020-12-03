# Hovatrace

Hovatrace is an annotated HTML trace generator for [HOVALAAG](http://silverspaceship.com/hovalaag/) implemented in Python 3. Over the built-in HTML trace it features

* Original assembler code overlayed with register state trace
* Syntax highlighting
* Jumps not taken are marked
* Constants shown as either (un)signed decimal, hex, or binary
* Tooltips for any constant showing it in all those number bases
* Comments (preceding and same-line) are associated with statements and used to annotate the trace
* Dark/light theme

Hovatrace takes as input a code file and a HOVALAAG-generated trace (usually `log.html`) and outputs another HTML file. Use `--help` for full documentation. `hovalag.py` can also be imported as a module providing code parsing/tokenization and trace parsing as an API. 
```
$ head 3.vasm
    B=20
    C=B,B=2047
    A=B
    D=A,B=-2048
loop:
; Pre: MAX in B, MIN in D.
    A=IN1

$ ./hoval 3 3.vasm -t 1
[...]

$ python hovatrace.py -c 3.vasm -fo output.html
```
This produces
<img src="https://github.com/jeweg/hovatrace/raw/main/sample.png">

(The sample shows an intentionally inefficient solution for problem 3 so as to not spoil much)