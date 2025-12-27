#![recursion_limit = "32768"]

use lisp_in_types::*;

defkey!(ASSOC, N0);
defkey!(KEY, N1);
defkey!(LS, N2);
defkey!(TAG, N3);
defkey!(RETURN, N4);
defkey!(CELL, N5);
defkey!(X, N6);
type Assoc = Defun<
    ASSOC,
    Cons<KEY, Cons<LS, Nil>>,
    lisp_in_types::CallEC<
        TAG,
        lisp_in_types::Lambda<
            Cons<RETURN, Nil>,
            lisp_in_types::If<
                lisp_in_types::Equalp<lisp_in_types::Var<LS>, lisp_in_types::Nil>,
                Call<lisp_in_types::Var<RETURN>, Cons<lisp_in_types::Nil, Nil>>,
                Let<
                    BCons<CELL, lisp_in_types::Car<lisp_in_types::Var<LS>>, BNil>,
                    lisp_in_types::If<
                        lisp_in_types::Eq<
                            lisp_in_types::Car<lisp_in_types::Var<CELL>>,
                            lisp_in_types::Var<KEY>,
                        >,
                        Call<lisp_in_types::Var<RETURN>, Cons<lisp_in_types::Var<CELL>, Nil>>,
                        Call<
                            lisp_in_types::Var<RETURN>,
                            Cons<
                                Call<
                                    lisp_in_types::Var<ASSOC>,
                                    Cons<
                                        lisp_in_types::Var<KEY>,
                                        Cons<lisp_in_types::Cdr<lisp_in_types::Var<LS>>, Nil>,
                                    >,
                                >,
                                Nil,
                            >,
                        >,
                    >,
                >,
            >,
        >,
    >,
>;
type GlobalEnv = <Assoc as EvalForm<ANil, ANil>>::GlobalOut;
type Body = expr!(
    (let ((X (cons (cons 1 10) (cons (cons 2 20) (cons (cons 3 30) nil)))))
        (ASSOC 2 X)));

type Result = EvalValue<Body, GlobalEnv, ANil>;

fn main() {
    println!("{:?}", <Result as ToRtValue>::to_rt());
}
