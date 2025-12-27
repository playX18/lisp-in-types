#![recursion_limit = "32768"]
use lisp_in_types::*;

fn main() {
    // AList example: ((x . 1) (y . 2))
    defkey!(SymX, N1);
    defkey!(SymY, N2);
    struct V1;
    struct V2;

    type Env = alist_t!((SymX => V1), (SymY => V2));
    type LookupX = <Env as Lookup<SymX>>::Output;
    type LookupY = <Env as Lookup<SymY>>::Output;
    defkey!(SymZ, N3);
    type LookupZ = <Env as Lookup<SymZ>>::Output;

    // Compile-time type checks.
    assert_same::<LookupX, Found<V1>>();
    assert_same::<LookupY, Found<V2>>();
    assert_same::<LookupZ, NotFound>();

    // Eval demo: (defun add (x y) (+ x y)) then (add 2 3) => 5
    defkey!(SymAdd, N4);

    type DefAdd = expr!((defun SymAdd (SymX SymY) (+ SymX SymY))); //defun!(SymAdd, (SymX, SymY), plus!(var!(SymX), var!(SymY)));
    type Global0 = ANil;
    type Lex0 = ANil;
    type Global1 = <DefAdd as EvalForm<Global0, Lex0>>::GlobalOut;

    type AddCall = expr!((SymAdd 2 3)); // call!(var!(SymAdd), lit!(N2), lit!(N3));
    type Result = EvalValue<AddCall, Global1, Lex0>;
    assert_same::<Result, N5>();

    println!("(add 2 3) => {:?}", <Result as ToRtValue>::to_rt());

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

    // Same call, but via Lisp-style `(apply add '(2 3))`.
    type Args23 = list_expr!(lit!(N2), lit!(N3));
    type AddApply = apply!(var!(SymAdd), Args23);
    type Result2 = EvalValue<AddApply, Global1, Lex0>;
    assert_same::<Result2, N5>();

    // New nodes demo.
    type TwoTimesThree = EvalValue<mul!(lit!(N2), lit!(N3)), Global1, Lex0>;
    assert_same::<TwoTimesThree, N6>();

    type TwoLtThree = EvalValue<lt!(lit!(N2), lit!(N3)), Global1, Lex0>;
    type ThreeGtTwo = EvalValue<gt!(lit!(N3), lit!(N2)), Global1, Lex0>;
    type TwoEqTwo = EvalValue<eq_!(lit!(N2), lit!(N2)), Global1, Lex0>;
    type TwoEqThree = EvalValue<eq_!(lit!(N2), lit!(N3)), Global1, Lex0>;
    assert_same::<TwoLtThree, True>();
    assert_same::<ThreeGtTwo, True>();
    assert_same::<TwoEqTwo, True>();
    assert_same::<TwoEqThree, False>();

    type IfExample = EvalValue<if_!(eq_!(lit!(N2), lit!(N2)), lit!(N10), lit!(N20)), Global1, Lex0>;
    assert_same::<IfExample, N10>();

    // Let demo: lexical binding shadows globals.
    // (let ((x 10)) (+ x 1)) => 11
    type LetExpr = let_!((SymX = lit!(N10)), plus!(var!(SymX), lit!(N1)));
    type LetResult = EvalValue<LetExpr, Global1, Lex0>;
    assert_same::<LetResult, N11>();

    // Lambda capture demo: `x` is captured from the surrounding lexical env.
    // (let ((x 10)) ((lambda (y) (+ x y)) 3)) => 13
    type CaptureExpr = let_!(
        (SymX = lit!(N10)),
        call!(lambda!((SymY), plus!(var!(SymX), var!(SymY))), lit!(N3))
    );
    type CaptureResult = EvalValue<CaptureExpr, Global1, Lex0>;
    assert_same::<CaptureResult, N13>();

    // set! demo: update an existing lexical binding.
    // (let ((x 10)) (set! x 20 x)) => 20
    type SetLexExpr = let_!((SymX = lit!(N10)), set_!((SymX = lit!(N20)), var!(SymX)));
    type SetLexResult = EvalValue<SetLexExpr, Global1, Lex0>;
    assert_same::<SetLexResult, N20>();

    // set! demo: if not in lex, update/add in global (for the remainder of the expression).
    // (set! g 7 (set! g 8 g)) => 8
    defkey!(SymG, N9);
    type SetGlobalExpr = set_!((SymG = lit!(N7)), set_!((SymG = lit!(N8)), var!(SymG)));
    type SetGlobalResult = EvalValue<SetGlobalExpr, Global1, Lex0>;
    assert_same::<SetGlobalResult, N8>();

    // Factorial as a defun (recursive).
    defkey!(SymFac, N7);
    defkey!(SymN, N8);

    type DefFac = expr!((defun SymFac (SymN) (if (= SymN 0) 1 (* SymN (SymFac (- SymN 1))))));

    type Global2 = <DefFac as EvalForm<Global1, Lex0>>::GlobalOut;
    type Fac5 = EvalValue<expr!((SymFac 5)), Global2, Lex0>;
    assert_same::<Fac5, N120>();

    println!("(fac 5) => {:?}", <Fac5 as ToRtValue>::to_rt());

    type ConsExpr = EvalValue<expr!((cons 1 2)), Global1, Lex0>;

    // begin demo: (begin 1 2 3) => 3
    type BeginExpr = EvalValue<expr!((begin 1 2 533)), Global1, Lex0>;
    assert_same::<BeginExpr, N533>();

    // begin + defun demo: (begin (defun foo () 42) (foo)) => 42
    defkey!(SymFoo, N42);
    type BeginDefunCall = EvalValue<expr!((begin (defun SymFoo () 1) (SymFoo))), Global1, Lex0>;
    assert_same::<BeginDefunCall, N1>();

    println!("(cons 1 2) => {:?}", <ConsExpr as ToRtValue>::to_rt());
    println!("(begin 1 2 533) => {:?}", <BeginExpr as ToRtValue>::to_rt());
    println!(
        "(begin (defun foo () 1) (foo)) => {:?}",
        <BeginDefunCall as ToRtValue>::to_rt()
    );

    defkey!(Assq, N0);
    defkey!(LIST, N1);
    defkey!(KEY, N2);
    type AssqDef = expr!(
    (defun Assq (KEY LIST)
        (if (equalp LIST nil)
            nil
            (if (equalp (car (car LIST)) KEY)
                (car LIST)
                (apply Assq (cons KEY (cons (cdr LIST) nil)))))));
    type GlobalAssq = <AssqDef as EvalForm<ANil, ANil>>::GlobalOut;
    type AssqResult = EvalValue<
        expr!(
            (Assq 2 (cons (cons 1 10) (cons (cons 2 20) (cons (cons 3 30) nil))))),
        GlobalAssq,
        ANil,
    >;

    println!(
        "(assq 2 (cons (cons 1 10) (cons (cons 2 20) (cons (cons 3 30) nil)))) => {:?}",
        <AssqResult as ToRtValue>::to_rt()
    );
}

defkey!(SymFac, N0);
defkey!(SymN, N1);
defkey!(SymAcc, N2);

type Fac = expr!(
    (defun SymFac (SymN SymAcc)
        (if (= SymN 0)
            SymAcc
            (SymFac (- SymN 1) (* SymN SymAcc)))));

type GlobalFac = <Fac as EvalForm<ANil, ANil>>::GlobalOut;
type FacOf5 = EvalValue<expr!((SymFac 5 1)), GlobalFac, ANil>;
const _: () = assert_same::<FacOf5, N120>();

defkey!(SLength, N0);

type LengthFunc = expr!(
    (defun SLength (Lst)
        (if (equalp Lst nil)
            0
            (+ 1 (SLength (cdr Lst))))));

type GlobalLength = <LengthFunc as EvalForm<ANil, ANil>>::GlobalOut;
type LengthOfList = EvalValue<expr!((SLength (cons 1 (cons 2 (cons 3 nil))))), GlobalLength, ANil>;
const _: () = assert_same::<LengthOfList, N3>();

defkey!(Map, N0);
defkey!(Func, N1);
defkey!(Lst, N2);
defkey!(Tmp, N3);
defkey!(One, N4);

type MapFunc = expr!(
    (defun Map (Func Lst)
        (if (equalp Lst nil)
            nil
            (cons
                (apply Func (cons (car Lst) nil))
                (Map Func (cdr Lst))))));

type GlobalMap = <MapFunc as EvalForm<ANil, ANil>>::GlobalOut;
type MapResult = EvalValue<
    expr!(
    (let ((One 1))
        (Map (lambda (Tmp) (+ One Tmp)) (cons 1 (cons 2 (cons 3 nil)))))),
    GlobalMap,
    ANil,
>;

const _: () = assert_same::<MapResult, Cons<N2, Cons<N3, Cons<N4, Nil>>>>();
