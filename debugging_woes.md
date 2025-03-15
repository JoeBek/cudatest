

The following details my adventure into debugging this cuda program...


# debugging woes

Disable caps lock if you use vim


Integral image not working (I think)
potential issues:
1. continuity problem (cv mat) - not it
2. reading it wrong  - BINGO
3. type issue - not it 
3.5 casting issue
4. allocation issue 
5. GPU sync issue - not it

It was fucking reading wrong...
goodbye 3 hours but I guess I got a little better at debugging...
It was made much worse by the fact that everything was coming up zero and I thought I could de-reference device pointer.

you cannot view GPU memory in cuda-gdb. Thanks chat. Idiot. AI is never taking over.

updates: 
img is being copied into cuda correctly.

The first fucking row and column is zero-padded.

At first I iterated through the first few elements (zero padded row).
Then I was like 'well, if i just iterate the rows, the integral image must be nonzero'
so I iterated (0, row, 2 * row) which is exactly the zero-padded column. Fuck this stupid game.. JK I love it.

But I learned some useful things. 1. you cant check device memory directly ever, 2. all the pair and pointer passing I did is fine. 3. I'm faster at gdb.

I'm gonna start debugging the actual problem here momentarily





# gdb stuff

this is acutally just more gdb grinding

```
gdb overall function

1. execution control
2. breakpoints
3. Inspection and modification
4. stack analysis
5. debugging symbols and source code
6. core dump analysis
7. remote debugging
8. scripting and automation
9. disassembly
10. Other useful things
```

```
execution control

starting and stopping programs

run | r - runs program
start - runs program and breaks at entrypoint
quit | q - quits gdb

continue | c - run from current point 
finish | f - run until current function returns

stepping through execution

step | s - step into
next | n - step over
stepi | si - step into machine instruction 
nexti | ni - step over machine instruction

```



```cpp
breakpoints

break <location>
location ::= <function name> | <file name>:<line_number> | *<address>


conditional breakpoint:

break <location> if <condition>

temporary breakpoint (gone after one hit)

tbreak <location>

hardware breakpoints

hbreak <location>

watchpoints (when a variable's value changes)

watch <variable>

catchpoints (signal or exception)

catch <event> 


you can find breakpoint with 

info <breakpoint>

delete/toggle them with 

delete <breakpoint_id>

disable <breakpoint_id>

enable <breakpoint_id>

you can configure the behavior of expressions that evalute at multiple spots with

set multiple-symbols all | ask | cancel
```

```
inspection and modification

info locals - show variable info in local scope

info args - show variable arguments

print | p <variable> - prints variable

display <variable> | <expr> - prints after every step

info registers - display cpu register contents

info frame - display information about stack frame






set <var> | <reg> - sets the value (wtf)



```





