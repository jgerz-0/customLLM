**Detailed Blueprint for a Classic BASIC Interpreter in C**

Below is a comprehensive, iterative plan to build a line-by-line BASIC interpreter in C. The project is broken down into broad phases, each subdivided into smaller chunks, and then further refined until each step feels manageable yet meaningful.

---

## High-Level Phases

1. **Core Infrastructure**
    1. **Tokenization**
    2. **Line Storage & Parsing**
    3. **Symbol Table & Variable Management**
2. **Interpreter Execution & Control Flow**
    1. **Line-by-Line Execution Loop**
    2. **Basic Control Structures (GOTO, GOSUB/RETURN, IF/THEN, FOR/NEXT)**
3. **Expanding Built-In BASIC Features**
    1. **Expressions & Built-In Functions**
    2. **String Operations & Dynamic Typing**
    3. **File I/O**
    4. **Sound & Graphics Hooks**
4. **Garbage Collection**
    1. **Mark-and-Sweep Implementation**
    2. **Integration & Testing**
5. **Debugger & REPL Enhancements**
    1. **Debugger Commands (BREAK, STEP, WATCH, LIST)**
    2. **REPL Integration & Error Handling**
    3. **State Save/Load**
6. **Final Integration & Testing**

---

## Breaking Down the Phases into Iterative Chunks

### Phase 1: Core Infrastructure

- **Chunk 1.1: Project Setup & Basic Skeleton** ✅
    - Create a basic C project with a main entry point (`main.c`).
    - Outline data structures for tokens, lines, and variables in header files.

- **Chunk 1.2: Tokenization** ✅
    - Write a tokenizer that reads one line of text and produces a list of tokens.
    - Support numbers, string literals, keywords (`PRINT`, `GOTO`, etc.), symbols (`=`, `+`, etc.).

- **Chunk 1.3: Line Storage & Parsing** ✅
    - Create a structure to store all lines by line number.
    - Parse each tokenized line to associate line numbers with their tokens.
    - Verify that line numbers are sorted/managed for the program.

- **Chunk 1.4: Symbol Table & Variable Management** ⚠️ (Partially Complete)
    - Implement a simple symbol table for variable lookup.
    - Start with an array or hash map to store variable names and their values.
    - Prepare to allow dynamic types (string vs numeric) later.
    - TODO: Add a test function in main.c to demonstrate variable storage and retrieval.

### Phase 2: Interpreter Execution & Control Flow

- **Chunk 2.1: Execution Loop (Line-by-Line)** ✅
    - Implement a loop that fetches the current line, executes it, and moves to the next.
    - Provide a program counter that updates based on normal flow or a GOTO jump.

- **Chunk 2.2: Control Structures: GOTO, GOSUB/RETURN, IF/THEN, FOR/NEXT**
    - Add logic for jumping to a line number with GOTO.
    - Implement a call stack for GOSUB/RETURN.
    - Introduce basic IF/THEN branching.
    - Implement FOR/NEXT loops with an internal stack or variable checks.

### Phase 3: Expanding Built-In BASIC Features

- **Chunk 3.1: Expressions & Built-In Functions**
    - Parse expressions (arithmetic, string concatenation, etc.).
    - Implement built-in functions like `RND`, `LEN`, `LEFT$`, `RIGHT$`, etc.

- **Chunk 3.2: String Operations & Dynamic Typing**
    - Extend variable storage to handle both numeric and string types.
    - Allow type conversion where appropriate (classic BASIC style).

- **Chunk 3.3: File I/O**
    - Support `OPEN`, `INPUT#`, `PRINT#`, `CLOSE`.
    - Handle file pointers in the symbol table or a dedicated structure.

- **Chunk 3.4: Sound & Graphics Hooks**
    - Implement `BEEP`/`SOUND`.
    - Create stubs or modular hooks for graphics commands (e.g., `LINE`, `CIRCLE`).

### Phase 4: Garbage Collection

- **Chunk 4.1: Mark-and-Sweep Implementation**
    - Add a mechanism to track live allocations for variables/strings.
    - Implement a mark phase that traverses active references.
    - Implement a sweep phase that frees unreferenced memory.

- **Chunk 4.2: Integration & Testing**
    - Integrate the garbage collector with the interpreter loop.
    - Test with programs that heavily use strings, arrays, etc.

### Phase 5: Debugger & REPL Enhancements

- **Chunk 5.1: Debugger Commands**
    - Implement `LIST`, `BREAK`, `STEP`, `WATCH`.
    - Track breakpoints using line numbers or conditions.

- **Chunk 5.2: REPL Integration & Error Handling**
    - Allow on-the-fly editing of lines in memory (like old BASIC).
    - Improve error handling for syntax/runtime errors.

- **Chunk 5.3: State Save/Load**
    - Serialize program lines, variables, and current state to a file.
    - Provide a command to load the state and resume execution.

### Phase 6: Final Integration & Testing

- **Chunk 6.1: Comprehensive Testing**
    - Create a suite of BASIC example programs for unit testing.
    - Stress test the interpreter with large scripts.

- **Chunk 6.2: Optimization & Cleanup**
    - Profile performance, optimize critical code paths.
    - Clean up the API for future extension.

---

## Breaking Each Chunk Into Smaller Steps

Below we iterate further, ensuring each chunk is small enough for safe implementation while still delivering progress. (We'll just illustrate for a subset; you'd apply the same pattern to every chunk.)

### Example: Chunk 1.2 (Tokenization) Detailed Breakdown

1. **Create a `token.h` to define token types** (keywords, identifiers, numbers, strings, operators). ✅
2. **Create a `tokenizer.c`** with functions like `init_tokenizer()`, `get_next_token()`, etc. ✅
3. **Implement a function to read a line of text** from standard input or file. ✅
4. **Split line into raw tokens** using whitespace as a delimiter. ✅
5. **Identify token type** (number, string, keyword, symbol). ✅
6. **Store tokens** in a dynamic array or linked list. ✅
7. **Test the tokenizer** with a few sample lines (e.g., `10 PRINT "HELLO"`). ✅

### Example: Chunk 2.1 (Execution Loop) Detailed Breakdown

1. **Create an `execute_line()` function** that takes a tokenized line.
2. **Initialize a program counter** to the first line in memory.
3. **Fetch the line** corresponding to the current program counter.
4. **Call `execute_line()`** with that line's tokens.
5. **By default, increment program counter** to the next line.
6. **Handle end-of-program** (no next line or `END` statement).
7. **Test** with a simple program that only has `PRINT` statements.

### Example: Chunk 4.1 (Mark-and-Sweep) Detailed Breakdown

1. **Design reference-tracking** in your variable structure (strings, arrays, etc.).
2. **Write `gc_mark()`** that marks all currently accessible variables.
3. **Write `gc_sweep()`** that frees memory blocks not marked.
4. **Integrate a call** to `gc_mark()` and `gc_sweep()` at appropriate intervals.
5. **Test** by intentionally creating memory cycles and verifying no crashes.

---

## Series of Prompts for Code-Generation LLM

Below are example prompts you could feed into a code-generation system. Each one assumes the code from previous steps is already in place, ensuring no orphaned code. Format them as you see fit; here we use Markdown code fences.

---

### Prompt 1: Project Skeleton and Data Structures 

```
Write C code for the initial project skeleton of a BASIC interpreter. 
Include:
- main.c that prints a greeting and sets up the structure for the interpreter.
- A header file "basic.h" defining basic data structures for Tokens, Lines, and a SymbolTable placeholder.
- Implementation files that include "basic.h" but otherwise do minimal work.
```

---

### Prompt 2: Tokenizer 

```
Extend the project by creating a tokenizer. 
Add:
- token.h to define an enum of token types (NUMBER, STRING, KEYWORD, SYMBOL, etc.).
- tokenizer.c to implement:
   init_tokenizer(),
   get_next_token(),
   a helper function tokenize_line() that accepts a C string and returns an array/list of tokens.
- Minimal testing in main.c or a separate test file that reads a single line and prints out tokens.
```

---

### Prompt 3: Line Storage & Parsing 

```
Add a line manager to store BASIC program lines by line number.
- line_manager.c with:
   add_line(int line_number, Token *tokens),
   get_line(int line_number),
   a data structure to store lines in a sorted manner (like a simple array or tree).
- Modify main.c to read multiple lines until EOF, tokenize them, and store them in line_manager.
- Print out the stored lines as a test.
```

---

### Prompt 4: Symbol Table & Variable Management 

```
Create a simple symbol table for variables in symbol_table.c. 
It should:
- Store variables by name as strings.
- Support different data types (initially just integer or float).
- Provide set_variable(const char *name, Value val) and get_variable(const char *name).
Integrate it into the main program but do not implement expression evaluation yet. 
Just show that we can store and retrieve variable values.
```

---

### Prompt 5: Execution Loop (Line-by-Line) 

```
Implement the core execution loop in execution.c:
- Add an execute_program() function that starts from the lowest line number and goes through lines.
- Each line calls execute_line(Line *line), which for now just prints "Executing line X" as a placeholder.
- After executing a line, move on to the next. 
- Stop if there are no more lines. 
Demonstrate with a small test program in main.c.
```

---

### Prompt 6: Control Structures (GOTO, GOSUB, IF/THEN, FOR/NEXT)

```
Enhance execute_line() to handle:
- GOTO <line_number>
- GOSUB <line_number> and RETURN
- IF <expr> THEN <line_number>
- FOR <var>=start TO end
- NEXT <var>
Implement the needed runtime structures (like a call stack). 
Demonstrate with a small BASIC program that includes GOTO, GOSUB, IF, and FOR loops.
```

---

### Prompt 7: Expressions & Built-In Functions

```
Add expression parsing and evaluation:
- expression.c with parse_expression(), evaluate_expression().
- Handle basic arithmetic (+, -, *, /), parentheses, and built-in functions (RND, LEN, LEFT$, RIGHT$).
Update execute_line() to parse expressions where needed (PRINT, IF, etc.).
Provide sample lines in main.c to test expression parsing.
```

---

### Prompt 8: String Operations & Dynamic Typing

```
Update the symbol table to store either numeric or string data.
Implement string creation, concatenation, and assignment.
Show classic BASIC style string variables (e.g., name$).
Include an example that manipulates strings with LEFT$, RIGHT$, and prints results.
```

---

### Prompt 9: File I/O

```
Add file I/O commands (OPEN, INPUT#, PRINT#, CLOSE).
Store file handles in a file manager or in the symbol table keyed by a file number or file variable.
Test by reading lines from a file and printing them out, then printing lines to a file.
```

---

### Prompt 10: Sound & Graphics Hooks

```
Create placeholders for BEEP/SOUND commands that call platform-specific beep functions or do nothing on unsupported platforms.
Add placeholders for LINE, CIRCLE, and other graphics commands that currently just print "Drawing line/circle" to console.
Confirm compilation and run.
```

---

### Prompt 11: Garbage Collection (Mark-and-Sweep) 

```
Integrate a custom mark-and-sweep garbage collector:
- memory_manager.c with track_allocation(), mark(), sweep(), etc.
- Mark everything reachable from global structures (symbol table, lines).
- Sweep orphaned objects.
Add a test that artificially creates allocations and triggers a GC cycle.
```

---

### Prompt 12: Debugger Commands (BREAK, STEP, WATCH, LIST) 

```
Extend the interpreter with debugger support:
- Create a debugger.c to handle commands: LIST, BREAK <line>, STEP, WATCH <var>.
- Integrate into the REPL. 
Add a global debug mode to the execution loop, checking for breakpoints and stepping.
```

---

### Prompt 13: REPL Integration & Error Handling

```
Combine everything into a real REPL in main.c:
- Users can type lines or debugger commands.
- The interpreter stores lines, executes them, and handles errors gracefully.
Test by simulating a minimal interactive BASIC environment.
```

---

### Prompt 14: State Save/Load

```
Implement state serialization:
- Save all lines, current program counter, symbol table data, etc. to a file.
- Load them back, resuming from the same point.
Demonstrate with a small script that saves state in the middle of a FOR loop, then loads and resumes.
```

---

### Prompt 15: Final Integration & Testing ✅

```
Perform final integration and QA:
- Merge all modules into a cohesive program.
- Clean up memory leaks, finalize GC usage.
- Test with real BASIC programs for correctness and performance.
```

---

## Conclusion

This blueprint offers a clear roadmap with progressively smaller tasks. Each prompt builds on the last. By following these prompts in order, you'll create a robust classic BASIC interpreter in C without leaving any code orphaned or partially integrated.