# lisp-in-types

Lisp in Rust trait system. 

# Limitations

- Each symbol must be manually declared via `defkey!()` macro
- Numbers cannot be negative
- Numbers are only in range of 0..8192, you can modify build.rs to generate more natural numbers
but then you have to run with RUST_MIN_STACK increased.
- No `(defmacro ...)`
- No `eval`
- I have not tested it extensively.

# Features

- Recursive functions
- Global and lexical environments (via `let` bindings)
- Function calls
- `apply` properly working
- `call/ec` 

# Example

## Factorial: 

```rust 

    type DefFac = expr!(defun SymFac (SymN) (if (= SymN 0) 1 (* SymN (SymFac (- SymN 1)))));

    type Global2 = <DefFac as EvalForm<Global1, Lex0>>::GlobalOut;
    type Fac5 = EvalValue<expr!((SymFac 5)), Global2, Lex0>;
    assert_same::<Fac5, N120>();

    println!("(fac 5) => {:?}", <Fac5 as ToRtValue>::to_rt());
```

## call/ec (delimited)

```rust

    // call/ec demo (escape continuation, explicit tag):
    // (call/ec cc (lambda (k) (+ 1 (k 5)))) => 5
    defkey!(SymCC, N10);
    defkey!(SymK, N11);
    type CallECExpr = expr!((call/ec SymCC (lambda (SymK) (+ 1 (SymK 5)))));
    type CallECResult = EvalValue<CallECExpr, Global1, Lex0>;
    assert_same::<CallECResult, N5>();
    println!(
        "(call/ec cc (lambda (k) (+ 1 (k 5)))) => {:?}",
        <CallECResult as ToRtValue>::to_rt()
    );
```