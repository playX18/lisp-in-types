#![recursion_limit = "32768"]
//! Lisp In Traits
//!
//! Minimal Lisp interpreter implemented entirely at the type level using Rust's
//! trait system.
use core::marker::PhantomData;

macro_rules! list_t {
    () => {
        Nil
    };
    ($head:ty $(, $tail:ty)* $(,)?) => {
        Cons<$head, list_t!($($tail),*)>
    };
}

macro_rules! alist_t {
    () => {
        ANil
    };
    (($k:ty => $v:ty) $(, ($k2:ty => $v2:ty))* $(,)?) => {
        ACons<$k, $v, alist_t!($(($k2 => $v2)),*)>
    };
}

macro_rules! defkey {
    ($name:ident, $id:ty) => {
        struct $name;
        impl Key for $name {
            type Id = $id;
        }
    };
}

macro_rules! var {
    ($k:ty) => {
        Var<$k>
    };
}

macro_rules! lit {
    ($v:ty) => {
        Const<$v>
    };
}

macro_rules! plus {
    ($a:ty, $b:ty) => {
        Plus<$a, $b>
    };
}

macro_rules! call {
    ($f:ty $(, $arg:ty)* $(,)?) => {
        Call<$f, list_t!($($arg),*)>
    };
}

macro_rules! list_expr {
    ($($expr:ty),* $(,)?) => {
        ListExpr<list_t!($($expr),*)>
    };
}

macro_rules! apply {
    ($f:ty, $args_list:ty) => {
        ApplyExpr<$f, $args_list>
    };
}

macro_rules! lambda {
    (($($param:ty),* $(,)?), $body:ty) => {
        Lambda<list_t!($($param),*), $body>
    };
}

macro_rules! defun {
    ($name:ty, ($($param:ty),* $(,)?), $body:ty) => {
        Defun<$name, list_t!($($param),*), $body>
    };
}

macro_rules! if_ {
    ($cond:ty, $then_:ty, $else_:ty) => {
        If<$cond, $then_, $else_>
    };
}

macro_rules! eq_ {
    ($a:ty, $b:ty) => {
        Eq<$a, $b>
    };
}

macro_rules! lt {
    ($a:ty, $b:ty) => {
        Lt<$a, $b>
    };
}

macro_rules! gt {
    ($a:ty, $b:ty) => {
        Gt<$a, $b>
    };
}

macro_rules! mul {
    ($a:ty, $b:ty) => {
        Mul<$a, $b>
    };
}

macro_rules! binds_t {
    () => {
        BNil
    };
    (($k:ty = $v:ty) $(, ($k2:ty = $v2:ty))* $(,)?) => {
        BCons<$k, $v, binds_t!($(($k2 = $v2)),*)>
    };
}

macro_rules! let_ {
    (($($k:ty = $v:ty),* $(,)?), $body:ty) => {
        Let<binds_t!($(($k = $v)),*), $body>
    };
}

macro_rules! set_ {
    (($k:ty = $v:ty), $body:ty) => {
        SetBang<$k, $v, $body>
    };
}

/// Marker trait for type-level lists.
pub trait List {}

/// Marker trait for non-empty type-level lists (i.e. anything but `Nil`).
pub trait NonNilList: List {}

/// The empty list.
#[derive(Debug, Clone, Copy, Default)]
pub struct Nil;

impl Expr for Nil {}

impl List for Nil {}

/// A cons cell (pair) containing `H` (head) and `T` (tail).
///
/// This is a type-level node: it carries no runtime data.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cons<H, T>(PhantomData<(H, T)>);

impl<H: Expr, T: Expr> Expr for Cons<H, T> {}

impl<H, T: List> List for Cons<H, T> {}

impl<H, T: List> NonNilList for Cons<H, T> {}

pub struct Car<L>(PhantomData<L>);
pub struct Cdr<L>(PhantomData<L>);

impl<L> Expr for Car<L> {}
impl<L> Expr for Cdr<L> {}

/// Deep equality: compare numbers like `=` and lists/pairs recursively.
pub struct Equalp<A: Expr, B: Expr>(PhantomData<(A, B)>);

impl<A: Expr, B: Expr> Expr for Equalp<A, B> {}

/// eqp: type-level equality on expressions.
pub struct Eqp<A: Expr, B: Expr>(PhantomData<(A, B)>);

impl<A: Expr, B: Expr> Expr for Eqp<A, B> {}

/// Trait representing a cons cell.
///
/// Any non-empty list node can implement this to expose its `Head` and `Tail`.
pub trait ConsCell: List {
    type Head;
    type Tail: List;
}

impl<H, T: List> ConsCell for Cons<H, T> {
    type Head = H;
    type Tail = T;
}

/// Type-level length.
pub trait Length: List {
    const VALUE: usize;
}

impl Length for Nil {
    const VALUE: usize = 0;
}

impl<H, T> Length for Cons<H, T>
where
    T: Length + List,
{
    const VALUE: usize = 1 + T::VALUE;
}
pub trait Same<T> {}
impl<T> Same<T> for T {}

const fn assert_same<T, U>()
where
    T: Same<U>,
{
}

/// Type-level booleans.
pub trait BoolT {}
#[derive(Debug, Clone, Copy, Default)]
pub struct True;
#[derive(Debug, Clone, Copy, Default)]
pub struct False;
impl BoolT for True {}
impl BoolT for False {}

/// Type-level boolean AND.
pub trait BoolAnd<Rhs: BoolT>: BoolT {
    type Output: BoolT;
}

impl<Rhs: BoolT> BoolAnd<Rhs> for True {
    type Output = Rhs;
}

impl<Rhs: BoolT> BoolAnd<Rhs> for False {
    type Output = False;
}

/// Type-level deep equality for evaluated values.
pub trait EqualpVal<Rhs> {
    type Output: BoolT;
}

impl<A: Nat, B: Nat> EqualpVal<B> for A
where
    A: EqNat<B>,
{
    type Output = <A as EqNat<B>>::Output;
}

impl EqualpVal<Nil> for Nil {
    type Output = True;
}

impl<AH, AT, BH, BT> EqualpVal<Cons<BH, BT>> for Cons<AH, AT>
where
    AH: EqualpVal<BH>,
    AT: EqualpVal<BT>,
    <AH as EqualpVal<BH>>::Output: BoolAnd<<AT as EqualpVal<BT>>::Output>,
{
    type Output = <<AH as EqualpVal<BH>>::Output as BoolAnd<<AT as EqualpVal<BT>>::Output>>::Output;
}

impl<A: Nat> EqualpVal<Nil> for A {
    type Output = False;
}

impl<A: Nat, BH, BT> EqualpVal<Cons<BH, BT>> for A {
    type Output = False;
}

impl<B: Nat> EqualpVal<B> for Nil {
    type Output = False;
}

impl<BH, BT> EqualpVal<Cons<BH, BT>> for Nil {
    type Output = False;
}

impl<AH, AT> EqualpVal<Nil> for Cons<AH, AT> {
    type Output = False;
}

impl<AH, AT, B: Nat> EqualpVal<B> for Cons<AH, AT> {
    type Output = False;
}

/// Type-level naturals (Peano encoding).
pub trait Nat: Expr {}
#[derive(Debug, Clone, Copy, Default)]
pub struct Z;
#[derive(Debug, Clone, Copy, Default)]
pub struct S<N: Nat>(PhantomData<N>);
impl Nat for Z {}
impl Expr for Z {}
impl<N: Nat> Nat for S<N> {}
impl<N: Nat> Expr for S<N> {}

/// Runtime representation of a subset of type-level values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtValue {
    Nat(usize),
    Bool(bool),
    Nil,
    Cons(Box<RtValue>, Box<RtValue>),
    Key(usize),
}

/// Convert a type-level natural into a `usize`.
pub trait ToUsize: Nat {
    const USIZE: usize;
}

impl ToUsize for Z {
    const USIZE: usize = 0;
}

impl<N: Nat + ToUsize> ToUsize for S<N> {
    const USIZE: usize = 1 + N::USIZE;
}

/// Convert a type-level boolean into a `bool`.
pub trait ToBool: BoolT {
    const BOOL: bool;
}

impl ToBool for True {
    const BOOL: bool = true;
}

impl ToBool for False {
    const BOOL: bool = false;
}

/// Convert a type-level value into a runtime enum.

pub trait ToRtValue {
    fn to_rt() -> RtValue;
}

impl ToRtValue for Z {
    fn to_rt() -> RtValue {
        RtValue::Nat(<Z as ToUsize>::USIZE)
    }
}

impl<N: Nat + ToUsize> ToRtValue for S<N> {
    fn to_rt() -> RtValue {
        RtValue::Nat(<S<N> as ToUsize>::USIZE)
    }
}

impl ToRtValue for True {
    fn to_rt() -> RtValue {
        RtValue::Bool(<True as ToBool>::BOOL)
    }
}

impl ToRtValue for False {
    fn to_rt() -> RtValue {
        RtValue::Bool(<False as ToBool>::BOOL)
    }
}

/// Convert a type-level list of reifiable elements into a runtime vector.
pub trait ToRtList: List {
    fn to_vec() -> Vec<RtValue>;
}

impl ToRtList for Nil {
    fn to_vec() -> Vec<RtValue> {
        Vec::new()
    }
}

impl<H, T> ToRtList for Cons<H, T>
where
    T: List + ToRtList,
    H: ToRtValue,
{
    fn to_vec() -> Vec<RtValue> {
        let mut items = Vec::new();
        items.push(H::to_rt());
        items.extend(T::to_vec());
        items
    }
}

impl<H, T> ToRtValue for Cons<H, T>
where
    T: ToRtValue,
    H: ToRtValue,
{
    fn to_rt() -> RtValue {
        RtValue::Cons(Box::new(H::to_rt()), Box::new(T::to_rt()))
    }
}

impl ToRtValue for Nil {
    fn to_rt() -> RtValue {
        RtValue::Nil
    }
}

include!(concat!(env!("OUT_DIR"), "/generated_nats.rs"));

/// Type-level equality on `Nat`.
pub trait EqNat<Rhs: Nat>: Nat {
    type Output: BoolT;
}

impl EqNat<Z> for Z {
    type Output = True;
}

impl<N: Nat> EqNat<S<N>> for Z {
    type Output = False;
}

impl<N: Nat> EqNat<Z> for S<N> {
    type Output = False;
}

impl<N: Nat, M: Nat> EqNat<S<M>> for S<N>
where
    N: EqNat<M>,
{
    type Output = <N as EqNat<M>>::Output;
}

/// Type-level strict less-than on `Nat`.
pub trait LtNat<Rhs: Nat>: Nat {
    type Output: BoolT;
}

impl LtNat<Z> for Z {
    type Output = False;
}

impl<M: Nat> LtNat<S<M>> for Z {
    type Output = True;
}

impl<N: Nat> LtNat<Z> for S<N> {
    type Output = False;
}

impl<N: Nat, M: Nat> LtNat<S<M>> for S<N>
where
    N: LtNat<M>,
{
    type Output = <N as LtNat<M>>::Output;
}

/// Type-level strict greater-than on `Nat`.
pub trait GtNat<Rhs: Nat>: Nat {
    type Output: BoolT;
}

impl GtNat<Z> for Z {
    type Output = False;
}

impl<M: Nat> GtNat<S<M>> for Z {
    type Output = False;
}

impl<N: Nat> GtNat<Z> for S<N> {
    type Output = True;
}

impl<N: Nat, M: Nat> GtNat<S<M>> for S<N>
where
    N: GtNat<M>,
{
    type Output = <N as GtNat<M>>::Output;
}

/// A type-level key (symbol) identified by a type-level natural.
///
/// For `Lookup` to be well-defined, IDs should be unique within a given `AList`.
pub trait Key {
    type Id: Nat;
}

/// Wrapper that reifies a `Key` by its numeric ID.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyId<K: Key>(PhantomData<K>);

impl<K: Key> ToRtValue for KeyId<K>
where
    K::Id: ToUsize,
{
    fn to_rt() -> RtValue {
        RtValue::Key(<K::Id as ToUsize>::USIZE)
    }
}

pub trait EqKey<Rhs: Key>: Key {
    type Output: BoolT;
}

impl<K: Key, Rhs: Key> EqKey<Rhs> for K
where
    K::Id: EqNat<Rhs::Id>,
{
    type Output = <K::Id as EqNat<Rhs::Id>>::Output;
}

/// Marker trait for type-level association lists.
pub trait AList {}

/// Empty association list.
#[derive(Debug, Clone, Copy, Default)]
pub struct ANil;

impl AList for ANil {}

/// Association list node: maps `K` to `V`, followed by `T`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ACons<K: Key, V, T: AList>(PhantomData<(K, V, T)>);

impl<K: Key, V, T: AList> AList for ACons<K, V, T> {}

/// Concatenate two association lists by appending `Rhs` after `Self`.
///
/// Lookups will prefer earlier bindings in `Self` (i.e. left list shadows right).
pub trait ConcatAList<Rhs: AList>: AList {
    type Out: AList;
}

impl<Rhs: AList> ConcatAList<Rhs> for ANil {
    type Out = Rhs;
}

impl<K: Key, V, T: AList, Rhs: AList> ConcatAList<Rhs> for ACons<K, V, T>
where
    T: ConcatAList<Rhs>,
{
    type Out = ACons<K, V, <T as ConcatAList<Rhs>>::Out>;
}

/// Lookup result (type-level option).
#[derive(Debug, Clone, Copy, Default)]
pub struct Found<V>(PhantomData<V>);

#[derive(Debug, Clone, Copy, Default)]
pub struct NotFound;

/// Lookup key `K` in an `AList`.
pub trait Lookup<K: Key>: AList {
    type Output;
}

impl<K: Key> Lookup<K> for ANil {
    type Output = NotFound;
}

pub trait LookupDispatch<K: Key, Match>: AList {
    type Output;
}

impl<K: Key, K2: Key, V, T: AList> Lookup<K> for ACons<K2, V, T>
where
    K2: EqKey<K>,
    ACons<K2, V, T>: LookupDispatch<K, <K2 as EqKey<K>>::Output>,
{
    type Output = <ACons<K2, V, T> as LookupDispatch<K, <K2 as EqKey<K>>::Output>>::Output;
}

impl<K: Key, K2: Key, V, T: AList> LookupDispatch<K, True> for ACons<K2, V, T> {
    type Output = Found<V>;
}

impl<K: Key, K2: Key, V, T: AList> LookupDispatch<K, False> for ACons<K2, V, T>
where
    T: Lookup<K>,
{
    type Output = <T as Lookup<K>>::Output;
}

/// Update the first binding for `K` in an `AList` to `NewV`.
///
/// Returns an updated list plus a type-level boolean indicating whether a
/// binding was found.
pub trait SetInAList<K: Key, NewV>: AList {
    type Out: AList;
    type Found: BoolT;
}

impl<K: Key, NewV> SetInAList<K, NewV> for ANil {
    type Out = ANil;
    type Found = False;
}

pub trait SetInAListDispatch<K: Key, NewV, Match: BoolT>: AList {
    type Out: AList;
    type Found: BoolT;
}

impl<K: Key, K2: Key, NewV, V, T: AList> SetInAList<K, NewV> for ACons<K2, V, T>
where
    K2: EqKey<K>,
    ACons<K2, V, T>: SetInAListDispatch<K, NewV, <K2 as EqKey<K>>::Output>,
{
    type Out = <ACons<K2, V, T> as SetInAListDispatch<K, NewV, <K2 as EqKey<K>>::Output>>::Out;
    type Found = <ACons<K2, V, T> as SetInAListDispatch<K, NewV, <K2 as EqKey<K>>::Output>>::Found;
}

impl<K: Key, K2: Key, NewV, V, T: AList> SetInAListDispatch<K, NewV, True> for ACons<K2, V, T> {
    type Out = ACons<K2, NewV, T>;
    type Found = True;
}

impl<K: Key, K2: Key, NewV, V, T: AList> SetInAListDispatch<K, NewV, False> for ACons<K2, V, T>
where
    T: SetInAList<K, NewV>,
{
    type Out = ACons<K2, V, <T as SetInAList<K, NewV>>::Out>;
    type Found = <T as SetInAList<K, NewV>>::Found;
}

/// Update `K` in `Self` if present; otherwise add a new binding at the front.
pub trait SetGlobalOrAdd<K: Key, NewV>: AList {
    type Out: AList;
}

pub trait SetGlobalOrAddDispatch<K: Key, NewV, Found: BoolT>: AList {
    type Out: AList;
}

impl<G, K: Key, NewV> SetGlobalOrAdd<K, NewV> for G
where
    G: AList + SetInAList<K, NewV>,
    G: SetGlobalOrAddDispatch<K, NewV, <G as SetInAList<K, NewV>>::Found>,
{
    type Out = <G as SetGlobalOrAddDispatch<K, NewV, <G as SetInAList<K, NewV>>::Found>>::Out;
}

impl<G, K: Key, NewV> SetGlobalOrAddDispatch<K, NewV, True> for G
where
    G: AList + SetInAList<K, NewV>,
{
    type Out = <G as SetInAList<K, NewV>>::Out;
}

impl<G, K: Key, NewV> SetGlobalOrAddDispatch<K, NewV, False> for G
where
    G: AList,
{
    type Out = ACons<K, NewV, G>;
}

/// Marker trait for type-level expressions.
pub trait Expr {}

/// Marker trait for top-level forms.
pub trait Form {}

/// A type-level lambda expression: `(lambda (params...) body)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lambda<Params: List, Body>(PhantomData<(Params, Body)>);

impl<Params: List, Body> Expr for Lambda<Params, Body> {}

pub trait LambdaExpr: Expr {
    type Params: List;
    type Body;
}

impl<Params: List, Body> LambdaExpr for Lambda<Params, Body> {
    type Params = Params;
    type Body = Body;
}

/// A type-level function definition: `(defun name (params...) body)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Defun<Name, Params: List, Body>(PhantomData<(Name, Params, Body)>);

impl<Name, Params: List, Body> Form for Defun<Name, Params, Body> {}

pub trait DefunForm: Form {
    type Name;
    type Params: List;
    type Body;
}

impl<Name, Params: List, Body> DefunForm for Defun<Name, Params, Body> {
    type Name = Name;
    type Params = Params;
    type Body = Body;
}

/// A type-level closure value produced by evaluating a `Lambda`.
///
/// It captures the lexical environment at definition time.
#[derive(Debug, Clone, Copy, Default)]
pub struct Closure<CEnv: AList, Params: List, Body>(PhantomData<(CEnv, Params, Body)>);

/// A recursive closure used for `defun` so the function can reference itself.
///
/// This avoids needing a self-referential environment type: when applying, we
/// evaluate the body in an env that includes a binding from `Name` to this
/// `RecClosure`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RecClosure<Name: Key, Global: AList, Lex: AList, Params: List, Body>(
    PhantomData<(Name, Global, Lex, Params, Body)>,
);

/// Variable reference.
#[derive(Debug, Clone, Copy, Default)]
pub struct Var<K: Key>(PhantomData<K>);
impl<K: Key> Expr for Var<K> {}

/// Constant value.
#[derive(Debug, Clone, Copy, Default)]
pub struct Const<V>(PhantomData<V>);
impl<V> Expr for Const<V> {}

/// Function application: `(f arg0 arg1 ...)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Call<F: Expr, Args: List>(PhantomData<(F, Args)>);
impl<F: Expr, Args: List> Expr for Call<F, Args> {}

/// List literal expression.
///
/// The list `L` is interpreted as a list of expressions, which will be evaluated
/// into a list of values.
#[derive(Debug, Clone, Copy, Default)]
pub struct ListExpr<L: List>(PhantomData<L>);
impl<L: List> Expr for ListExpr<L> {}

/// Lisp-style apply: `(apply f args_list)`.
///
/// `args_list` is an expression that evaluates to a list of argument values.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApplyExpr<F: Expr, ArgsList: Expr>(PhantomData<(F, ArgsList)>);
impl<F: Expr, ArgsList: Expr> Expr for ApplyExpr<F, ArgsList> {}

/// Let bindings (let* semantics): sequentially bind keys to evaluated values.
pub trait Bindings {}

#[derive(Debug, Clone, Copy, Default)]
pub struct BNil;
impl Bindings for BNil {}

#[derive(Debug, Clone, Copy, Default)]
pub struct BCons<K: Key, V: Expr, T: Bindings>(PhantomData<(K, V, T)>);
impl<K: Key, V: Expr, T: Bindings> Bindings for BCons<K, V, T> {}

/// `(let ((k0 v0) (k1 v1) ...) body)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Let<B: Bindings, Body: Expr>(PhantomData<(B, Body)>);
impl<B: Bindings, Body: Expr> Expr for Let<B, Body> {}

/// A sequencing / assignment expression.
///
/// `(set! k v body)` evaluates `v`, then updates `k` in the lexical env if it
/// exists; otherwise updates (or inserts) `k` in the global env; then evaluates
/// `body` in the updated environment.
#[derive(Debug, Clone, Copy, Default)]
pub struct SetBang<K: Key, V: Expr, Body: Expr>(PhantomData<(K, V, Body)>);
impl<K: Key, V: Expr, Body: Expr> Expr for SetBang<K, V, Body> {}

/// Sequential evaluation: `(begin e0 e1 ... en)`.
///
/// Evaluates each item left-to-right and returns the value of the last item.
///
/// Items are usually expressions, but `begin` also supports top-level forms
/// like `defun` within the sequence; such forms update the *global*
/// environment for subsequent items in the same `begin`.
///
/// If the sequence is empty, it evaluates to `Nil`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Begin<Seq: List>(PhantomData<Seq>);
impl<Seq: List> Expr for Begin<Seq> {}

/// Integer addition on type-level naturals.
#[derive(Debug, Clone, Copy, Default)]
pub struct Plus<A: Expr, B: Expr>(PhantomData<(A, B)>);
impl<A: Expr, B: Expr> Expr for Plus<A, B> {}

/// Integer multiplication on type-level naturals.
#[derive(Debug, Clone, Copy, Default)]
pub struct Mul<A: Expr, B: Expr>(PhantomData<(A, B)>);
impl<A: Expr, B: Expr> Expr for Mul<A, B> {}

/// Integer subtraction on type-level naturals (saturating at zero).
#[derive(Debug, Clone, Copy, Default)]
pub struct Sub<A: Expr, B: Expr>(PhantomData<(A, B)>);
impl<A: Expr, B: Expr> Expr for Sub<A, B> {}

/// Numeric equality (on evaluated `Nat` values).
#[derive(Debug, Clone, Copy, Default)]
pub struct Eq<A: Expr, B: Expr>(PhantomData<(A, B)>);
impl<A: Expr, B: Expr> Expr for Eq<A, B> {}

/// Numeric less-than (on evaluated `Nat` values).
#[derive(Debug, Clone, Copy, Default)]
pub struct Lt<A: Expr, B: Expr>(PhantomData<(A, B)>);
impl<A: Expr, B: Expr> Expr for Lt<A, B> {}

/// Numeric greater-than (on evaluated `Nat` values).
#[derive(Debug, Clone, Copy, Default)]
pub struct Gt<A: Expr, B: Expr>(PhantomData<(A, B)>);
impl<A: Expr, B: Expr> Expr for Gt<A, B> {}

/// Conditional expression.
#[derive(Debug, Clone, Copy, Default)]
pub struct If<Cond: Expr, Then: Expr, Else: Expr>(PhantomData<(Cond, Then, Else)>);
impl<Cond: Expr, Then: Expr, Else: Expr> Expr for If<Cond, Then, Else> {}

/// Type-level addition on Peano naturals.
pub trait AddNat<Rhs: Nat>: Nat {
    type Output: Nat;
}

impl<M: Nat> AddNat<M> for Z {
    type Output = M;
}

impl<N: Nat, M: Nat> AddNat<M> for S<N>
where
    N: AddNat<M>,
{
    type Output = S<<N as AddNat<M>>::Output>;
}

/// Type-level multiplication on Peano naturals.
pub trait MulNat<Rhs: Nat>: Nat {
    type Output: Nat;
}

impl<M: Nat> MulNat<M> for Z {
    type Output = Z;
}

impl<N: Nat, M: Nat> MulNat<M> for S<N>
where
    N: MulNat<M>,
    M: AddNat<<N as MulNat<M>>::Output> + Nat,
{
    type Output = <M as AddNat<<N as MulNat<M>>::Output>>::Output;
}

/// Type-level subtraction on Peano naturals (saturating at zero).
pub trait SubNat<Rhs: Nat>: Nat {
    type Output: Nat;
}

impl<M: Nat> SubNat<M> for Z {
    type Output = Z;
}

impl<N: Nat> SubNat<Z> for S<N> {
    type Output = S<N>;
}

impl<N: Nat, M: Nat> SubNat<S<M>> for S<N>
where
    N: SubNat<M>,
{
    type Output = <N as SubNat<M>>::Output;
}

/// Control-flow result of evaluation.
///
/// This lets us implement escape continuations (and `call/ec`-style early exit).
#[derive(Debug, Clone, Copy, Default)]
pub struct Ok<V>(PhantomData<V>);

#[derive(Debug, Clone, Copy, Default)]
pub struct Escape<Tag, V>(PhantomData<(Tag, V)>);

pub trait Control {}
impl<V> Control for Ok<V> {}
impl<Tag, V> Control for Escape<Tag, V> {}

/// Extract the value from `Ok<V>`.
///
/// There is intentionally no impl for `Escape<..>`: letting an escape bubble
/// out of the context that should catch it becomes a compile-time error.
pub trait UnwrapOk: Control {
    type Value;
}

impl<V> UnwrapOk for Ok<V> {
    type Value = V;
}

/// Convenience alias for getting the plain value of an expression.
pub type EvalValue<E, Global, Lex> = <<E as Eval<Global, Lex>>::Value as UnwrapOk>::Value;

impl<V: ToRtValue> ToRtValue for Ok<V> {
    fn to_rt() -> RtValue {
        V::to_rt()
    }
}

/// Evaluates an expression in an environment.
///
/// The result is a type (a "value") in the associated type `Value`.
pub trait Eval<Global: AList, Lex: AList>: Expr {
    type Value: Control;
}

impl<Global: AList, Lex: AList, V> Eval<Global, Lex> for Const<V> {
    type Value = Ok<V>;
}

impl<Global: AList, Lex: AList> Eval<Global, Lex> for Nil {
    type Value = Ok<Nil>;
}

pub trait EvalCarDispatch<Global: AList, Lex: AList, Arg: Expr, C: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, Arg: Expr, Tag, V> EvalCarDispatch<Global, Lex, Arg, Escape<Tag, V>>
    for Car<Arg>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, Arg: Expr, V> EvalCarDispatch<Global, Lex, Arg, Ok<V>> for Car<Arg>
where
    V: ConsCell,
{
    type Out = Ok<<V as ConsCell>::Head>;
}

impl<Global: AList, Lex: AList, Arg: Expr> Eval<Global, Lex> for Car<Arg>
where
    Arg: Eval<Global, Lex>,
    Car<Arg>: EvalCarDispatch<Global, Lex, Arg, <Arg as Eval<Global, Lex>>::Value>,
{
    type Value =
        <Car<Arg> as EvalCarDispatch<Global, Lex, Arg, <Arg as Eval<Global, Lex>>::Value>>::Out;
}

pub trait EvalCdrDispatch<Global: AList, Lex: AList, Arg: Expr, C: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, Arg: Expr, Tag, V> EvalCdrDispatch<Global, Lex, Arg, Escape<Tag, V>>
    for Cdr<Arg>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, Arg: Expr, V> EvalCdrDispatch<Global, Lex, Arg, Ok<V>> for Cdr<Arg>
where
    V: ConsCell,
{
    type Out = Ok<<V as ConsCell>::Tail>;
}

impl<Global: AList, Lex: AList, Arg: Expr> Eval<Global, Lex> for Cdr<Arg>
where
    Arg: Eval<Global, Lex>,
    Cdr<Arg>: EvalCdrDispatch<Global, Lex, Arg, <Arg as Eval<Global, Lex>>::Value>,
{
    type Value =
        <Cdr<Arg> as EvalCdrDispatch<Global, Lex, Arg, <Arg as Eval<Global, Lex>>::Value>>::Out;
}

pub trait EvalConsDispatch<Global: AList, Lex: AList, H: Expr, T: Expr, CH: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, H: Expr, T: Expr, Tag, V>
    EvalConsDispatch<Global, Lex, H, T, Escape<Tag, V>> for Cons<H, T>
{
    type Out = Escape<Tag, V>;
}

pub trait EvalConsTailDispatch<Global: AList, Lex: AList, HVal, T: Expr, CT: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, HVal, T: Expr, Tag, V>
    EvalConsTailDispatch<Global, Lex, HVal, T, Escape<Tag, V>> for Cons<HVal, T>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, HVal, T: Expr, TVal>
    EvalConsTailDispatch<Global, Lex, HVal, T, Ok<TVal>> for Cons<HVal, T>
{
    type Out = Ok<Cons<HVal, TVal>>;
}

impl<Global: AList, Lex: AList, H: Expr, T: Expr> Eval<Global, Lex> for Cons<H, T>
where
    H: Eval<Global, Lex>,
    Cons<H, T>: EvalConsDispatch<Global, Lex, H, T, <H as Eval<Global, Lex>>::Value>,
{
    type Value =
        <Cons<H, T> as EvalConsDispatch<Global, Lex, H, T, <H as Eval<Global, Lex>>::Value>>::Out;
}

impl<Global: AList, Lex: AList, H: Expr, T: Expr, HVal>
    EvalConsDispatch<Global, Lex, H, T, Ok<HVal>> for Cons<H, T>
where
    T: Eval<Global, Lex>,
    Cons<HVal, T>: EvalConsTailDispatch<Global, Lex, HVal, T, <T as Eval<Global, Lex>>::Value>,
{
    type Out = <Cons<HVal, T> as EvalConsTailDispatch<
        Global,
        Lex,
        HVal,
        T,
        <T as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global: AList, Lex: AList, N: Nat> Eval<Global, Lex> for N {
    type Value = Ok<N>;
}

impl<Global, Lex, L> Eval<Global, Lex> for ListExpr<L>
where
    Global: AList,
    Lex: AList,
    L: List + EvalList<Global, Lex>,
{
    type Value = <L as EvalList<Global, Lex>>::Output;
}

pub trait UnwrapFound {
    type Value;
}

impl<V> UnwrapFound for Found<V> {
    type Value = V;
}

pub trait ResolveLookup<Global: AList, K: Key> {
    type Value;
}

impl<Global: AList, K: Key, V> ResolveLookup<Global, K> for Found<V> {
    type Value = V;
}

impl<Global: AList, K: Key> ResolveLookup<Global, K> for NotFound
where
    Global: Lookup<K>,
    <Global as Lookup<K>>::Output: UnwrapFound,
{
    type Value = <<Global as Lookup<K>>::Output as UnwrapFound>::Value;
}

impl<Global, Lex, K> Eval<Global, Lex> for Var<K>
where
    Global: AList,
    Lex: AList + Lookup<K>,
    K: Key,
    <Lex as Lookup<K>>::Output: ResolveLookup<Global, K>,
{
    type Value = Ok<<<Lex as Lookup<K>>::Output as ResolveLookup<Global, K>>::Value>;
}

impl<Global, Lex, Params, Body> Eval<Global, Lex> for Lambda<Params, Body>
where
    Global: AList,
    Lex: AList,
    Params: List,
    Body: Expr,
{
    type Value = Ok<Closure<Lex, Params, Body>>;
}

pub trait EvalList<Global: AList, Lex: AList>: List {
    type Output: Control;
}

impl<Global: AList, Lex: AList> EvalList<Global, Lex> for Nil {
    type Output = Ok<Nil>;
}

pub trait EvalListHeadDispatch<Global: AList, Lex: AList, H: Expr, T: List, CH: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, H: Expr, T: List, Tag, V>
    EvalListHeadDispatch<Global, Lex, H, T, Escape<Tag, V>> for Cons<H, T>
{
    type Out = Escape<Tag, V>;
}

pub trait EvalListTailDispatch<Global: AList, Lex: AList, HVal, T: List, CT: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, HVal, T: List, Tag, V>
    EvalListTailDispatch<Global, Lex, HVal, T, Escape<Tag, V>> for Cons<HVal, T>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, HVal, T: List, TVal: List>
    EvalListTailDispatch<Global, Lex, HVal, T, Ok<TVal>> for Cons<HVal, T>
{
    type Out = Ok<Cons<HVal, TVal>>;
}

impl<Global, Lex, H, T> EvalList<Global, Lex> for Cons<H, T>
where
    Global: AList,
    Lex: AList,
    H: Expr + Eval<Global, Lex>,
    T: List,
    Cons<H, T>: EvalListHeadDispatch<Global, Lex, H, T, <H as Eval<Global, Lex>>::Value>,
{
    type Output = <Cons<H, T> as EvalListHeadDispatch<
        Global,
        Lex,
        H,
        T,
        <H as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, H, T, HVal> EvalListHeadDispatch<Global, Lex, H, T, Ok<HVal>> for Cons<H, T>
where
    Global: AList,
    Lex: AList,
    H: Expr,
    T: List + EvalList<Global, Lex>,
    Cons<HVal, T>: EvalListTailDispatch<Global, Lex, HVal, T, <T as EvalList<Global, Lex>>::Output>,
{
    type Out = <Cons<HVal, T> as EvalListTailDispatch<
        Global,
        Lex,
        HVal,
        T,
        <T as EvalList<Global, Lex>>::Output,
    >>::Out;
}

/// Apply a function value to an evaluated argument list.
pub trait Apply<Global: AList, Args: List> {
    type Output: Control;
}

/// Bind `Params` (a list of keys) to `Args` (a list of values), extending an env.
pub trait Bind<Args: List, Env: AList>: List {
    type Out: AList;
}

impl<Env: AList> Bind<Nil, Env> for Nil {
    type Out = Env;
}

impl<PHead, PTail, AHead, ATail, Env> Bind<Cons<AHead, ATail>, Env> for Cons<PHead, PTail>
where
    PHead: Key,
    PTail: List + Bind<ATail, ACons<PHead, AHead, Env>>,
    ATail: List,
    Env: AList,
{
    type Out = <PTail as Bind<ATail, ACons<PHead, AHead, Env>>>::Out;
}

impl<Global, LexCap, Params, Body, Args> Apply<Global, Args> for Closure<LexCap, Params, Body>
where
    Global: AList,
    LexCap: AList,
    Params: List + Bind<Args, LexCap>,
    Args: List,
    Body: Expr,
    Body: Eval<Global, <Params as Bind<Args, LexCap>>::Out>,
{
    type Output = <Body as Eval<Global, <Params as Bind<Args, LexCap>>::Out>>::Value;
}

impl<Name, GlobalCall, GlobalCap, LexCap, Params, Body, Args> Apply<GlobalCall, Args>
    for RecClosure<Name, GlobalCap, LexCap, Params, Body>
where
    Name: Key,
    GlobalCall: AList,
    GlobalCap: AList,
    LexCap: AList + ConcatAList<GlobalCall>,
    Params: List,
    Args: List,
    // Make globals available as part of the lexical env (Lex ++ Global).
    Params: Bind<
            Args,
            ACons<
                Name,
                RecClosure<Name, GlobalCap, LexCap, Params, Body>,
                <LexCap as ConcatAList<GlobalCall>>::Out,
            >,
        >,
    Body: Expr,
    Body: Eval<
            GlobalCall,
            <Params as Bind<
                Args,
                ACons<
                    Name,
                    RecClosure<Name, GlobalCap, LexCap, Params, Body>,
                    <LexCap as ConcatAList<GlobalCall>>::Out,
                >,
            >>::Out,
        >,
{
    type Output = <Body as Eval<
        GlobalCall,
        <Params as Bind<
            Args,
            ACons<
                Name,
                RecClosure<Name, GlobalCap, LexCap, Params, Body>,
                <LexCap as ConcatAList<GlobalCall>>::Out,
            >,
        >>::Out,
    >>::Value;
}

pub trait EvalBinds<Global: AList, Lex: AList>: Bindings {
    type Output: Control;
}

impl<Global: AList, Lex: AList> EvalBinds<Global, Lex> for BNil {
    type Output = Ok<Lex>;
}

pub trait EvalBindsHeadDispatch<
    Global: AList,
    Lex: AList,
    K: Key,
    V: Expr,
    T: Bindings,
    CV: Control,
>
{
    type Out: Control;
}

impl<Global: AList, Lex: AList, K: Key, V: Expr, T: Bindings, Tag, X>
    EvalBindsHeadDispatch<Global, Lex, K, V, T, Escape<Tag, X>> for BCons<K, V, T>
{
    type Out = Escape<Tag, X>;
}

impl<Global, Lex, K, V, T, VVal> EvalBindsHeadDispatch<Global, Lex, K, V, T, Ok<VVal>>
    for BCons<K, V, T>
where
    Global: AList,
    Lex: AList,
    K: Key,
    V: Expr,
    T: Bindings + EvalBinds<Global, ACons<K, VVal, Lex>>,
{
    type Out = <T as EvalBinds<Global, ACons<K, VVal, Lex>>>::Output;
}

impl<Global, Lex, K, V, T> EvalBinds<Global, Lex> for BCons<K, V, T>
where
    Global: AList,
    Lex: AList,
    K: Key,
    V: Expr + Eval<Global, Lex>,
    BCons<K, V, T>: EvalBindsHeadDispatch<Global, Lex, K, V, T, <V as Eval<Global, Lex>>::Value>,
    T: Bindings,
{
    type Output = <BCons<K, V, T> as EvalBindsHeadDispatch<
        Global,
        Lex,
        K,
        V,
        T,
        <V as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, B, Body> Eval<Global, Lex> for Let<B, Body>
where
    Global: AList,
    Lex: AList,
    B: Bindings + EvalBinds<Global, Lex>,
    Body: Expr,
    Let<B, Body>: LetDispatch<Global, Lex, B, Body, <B as EvalBinds<Global, Lex>>::Output>,
{
    type Value = <Let<B, Body> as LetDispatch<
        Global,
        Lex,
        B,
        Body,
        <B as EvalBinds<Global, Lex>>::Output,
    >>::Out;
}

pub trait LetDispatch<Global: AList, Lex: AList, B: Bindings, Body: Expr, CBinds: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, B: Bindings, Body: Expr, Tag, V>
    LetDispatch<Global, Lex, B, Body, Escape<Tag, V>> for Let<B, Body>
{
    type Out = Escape<Tag, V>;
}

impl<Global, Lex, B, Body, Lex2> LetDispatch<Global, Lex, B, Body, Ok<Lex2>> for Let<B, Body>
where
    Global: AList,
    Lex: AList,
    B: Bindings,
    Lex2: AList,
    Body: Expr + Eval<Global, Lex2>,
{
    type Out = <Body as Eval<Global, Lex2>>::Value;
}

pub trait SetBangDispatch<Global: AList, Lex: AList, K: Key, V: Expr, Body: Expr, CV: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, K: Key, V: Expr, Body: Expr, Tag, X>
    SetBangDispatch<Global, Lex, K, V, Body, Escape<Tag, X>> for SetBang<K, V, Body>
{
    type Out = Escape<Tag, X>;
}

impl<Global, Lex, K, V, Body, Val> SetBangDispatch<Global, Lex, K, V, Body, Ok<Val>>
    for SetBang<K, V, Body>
where
    Global: AList,
    Lex: AList,
    K: Key,
    V: Expr,
    Body: Expr,
    Lex: SetInAList<K, Val>,
    SetBang<K, V, Body>:
        SetBangFoundDispatch<Global, Lex, K, Body, Val, <Lex as SetInAList<K, Val>>::Found>,
{
    type Out = <SetBang<K, V, Body> as SetBangFoundDispatch<
        Global,
        Lex,
        K,
        Body,
        Val,
        <Lex as SetInAList<K, Val>>::Found,
    >>::Out;
}

pub trait SetBangFoundDispatch<Global: AList, Lex: AList, K: Key, Body: Expr, Val, Found: BoolT> {
    type Out: Control;
}

impl<Global, Lex, K, V, Body, Val> SetBangFoundDispatch<Global, Lex, K, Body, Val, True>
    for SetBang<K, V, Body>
where
    Global: AList,
    Lex: AList + SetInAList<K, Val>,
    K: Key,
    V: Expr,
    Body: Expr + Eval<Global, <Lex as SetInAList<K, Val>>::Out>,
{
    type Out = <Body as Eval<Global, <Lex as SetInAList<K, Val>>::Out>>::Value;
}

impl<Global, Lex, K, V, Body, Val> SetBangFoundDispatch<Global, Lex, K, Body, Val, False>
    for SetBang<K, V, Body>
where
    Global: AList + SetGlobalOrAdd<K, Val>,
    Lex: AList,
    K: Key,
    V: Expr,
    Body: Expr + Eval<<Global as SetGlobalOrAdd<K, Val>>::Out, Lex>,
{
    type Out = <Body as Eval<<Global as SetGlobalOrAdd<K, Val>>::Out, Lex>>::Value;
}

impl<Global, Lex, K, V, Body> Eval<Global, Lex> for SetBang<K, V, Body>
where
    Global: AList,
    Lex: AList,
    K: Key,
    V: Expr + Eval<Global, Lex>,
    Body: Expr,
    SetBang<K, V, Body>: SetBangDispatch<Global, Lex, K, V, Body, <V as Eval<Global, Lex>>::Value>,
{
    type Value = <SetBang<K, V, Body> as SetBangDispatch<
        Global,
        Lex,
        K,
        V,
        Body,
        <V as Eval<Global, Lex>>::Value,
    >>::Out;
}

pub trait BinOpLeftDispatch<Global: AList, Lex: AList, A: Expr, B: Expr, CA: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, A: Expr, B: Expr, Tag, V>
    BinOpLeftDispatch<Global, Lex, A, B, Escape<Tag, V>> for Plus<A, B>
{
    type Out = Escape<Tag, V>;
}

pub trait PlusRightDispatch<Global: AList, Lex: AList, AVal: Nat, B: Expr, CB: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, Tag, V>
    PlusRightDispatch<Global, Lex, AVal, B, Escape<Tag, V>> for Plus<Const<AVal>, B>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, BVal: Nat>
    PlusRightDispatch<Global, Lex, AVal, B, Ok<BVal>> for Plus<Const<AVal>, B>
where
    AVal: AddNat<BVal>,
{
    type Out = Ok<<AVal as AddNat<BVal>>::Output>;
}

impl<Global, Lex, A, B, AVal> BinOpLeftDispatch<Global, Lex, A, B, Ok<AVal>> for Plus<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr,
    B: Expr + Eval<Global, Lex>,
    AVal: Nat,
    Plus<Const<AVal>, B>: PlusRightDispatch<Global, Lex, AVal, B, <B as Eval<Global, Lex>>::Value>,
{
    type Out = <Plus<Const<AVal>, B> as PlusRightDispatch<
        Global,
        Lex,
        AVal,
        B,
        <B as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, A, B> Eval<Global, Lex> for Plus<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr + Eval<Global, Lex>,
    B: Expr,
    Plus<A, B>: BinOpLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>,
{
    type Value =
        <Plus<A, B> as BinOpLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>>::Out;
}

pub trait MulLeftDispatch<Global: AList, Lex: AList, A: Expr, B: Expr, CA: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, A: Expr, B: Expr, Tag, V>
    MulLeftDispatch<Global, Lex, A, B, Escape<Tag, V>> for Mul<A, B>
{
    type Out = Escape<Tag, V>;
}

pub trait MulRightDispatch<Global: AList, Lex: AList, AVal: Nat, B: Expr, CB: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, Tag, V>
    MulRightDispatch<Global, Lex, AVal, B, Escape<Tag, V>> for Mul<Const<AVal>, B>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, BVal: Nat>
    MulRightDispatch<Global, Lex, AVal, B, Ok<BVal>> for Mul<Const<AVal>, B>
where
    AVal: MulNat<BVal>,
{
    type Out = Ok<<AVal as MulNat<BVal>>::Output>;
}

impl<Global, Lex, A, B, AVal> MulLeftDispatch<Global, Lex, A, B, Ok<AVal>> for Mul<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr,
    B: Expr + Eval<Global, Lex>,
    AVal: Nat,
    Mul<Const<AVal>, B>: MulRightDispatch<Global, Lex, AVal, B, <B as Eval<Global, Lex>>::Value>,
{
    type Out = <Mul<Const<AVal>, B> as MulRightDispatch<
        Global,
        Lex,
        AVal,
        B,
        <B as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, A, B> Eval<Global, Lex> for Mul<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr + Eval<Global, Lex>,
    B: Expr,
    Mul<A, B>: MulLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>,
{
    type Value =
        <Mul<A, B> as MulLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>>::Out;
}

pub trait SubLeftDispatch<Global: AList, Lex: AList, A: Expr, B: Expr, CA: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, A: Expr, B: Expr, Tag, V>
    SubLeftDispatch<Global, Lex, A, B, Escape<Tag, V>> for Sub<A, B>
{
    type Out = Escape<Tag, V>;
}

pub trait SubRightDispatch<Global: AList, Lex: AList, AVal: Nat, B: Expr, CB: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, Tag, V>
    SubRightDispatch<Global, Lex, AVal, B, Escape<Tag, V>> for Sub<Const<AVal>, B>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, BVal: Nat>
    SubRightDispatch<Global, Lex, AVal, B, Ok<BVal>> for Sub<Const<AVal>, B>
where
    AVal: SubNat<BVal>,
{
    type Out = Ok<<AVal as SubNat<BVal>>::Output>;
}

impl<Global, Lex, A, B, AVal> SubLeftDispatch<Global, Lex, A, B, Ok<AVal>> for Sub<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr,
    B: Expr + Eval<Global, Lex>,
    AVal: Nat,
    Sub<Const<AVal>, B>: SubRightDispatch<Global, Lex, AVal, B, <B as Eval<Global, Lex>>::Value>,
{
    type Out = <Sub<Const<AVal>, B> as SubRightDispatch<
        Global,
        Lex,
        AVal,
        B,
        <B as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, A, B> Eval<Global, Lex> for Sub<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr + Eval<Global, Lex>,
    B: Expr,
    Sub<A, B>: SubLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>,
{
    type Value =
        <Sub<A, B> as SubLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>>::Out;
}

pub trait EqLeftDispatch<Global: AList, Lex: AList, A: Expr, B: Expr, CA: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, A: Expr, B: Expr, Tag, V>
    EqLeftDispatch<Global, Lex, A, B, Escape<Tag, V>> for Eq<A, B>
{
    type Out = Escape<Tag, V>;
}

pub trait EqRightDispatch<Global: AList, Lex: AList, AVal: Nat, B: Expr, CB: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, Tag, V>
    EqRightDispatch<Global, Lex, AVal, B, Escape<Tag, V>> for Eq<Const<AVal>, B>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, BVal: Nat>
    EqRightDispatch<Global, Lex, AVal, B, Ok<BVal>> for Eq<Const<AVal>, B>
where
    AVal: EqNat<BVal>,
{
    type Out = Ok<<AVal as EqNat<BVal>>::Output>;
}

impl<Global, Lex, A, B, AVal> EqLeftDispatch<Global, Lex, A, B, Ok<AVal>> for Eq<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr,
    B: Expr + Eval<Global, Lex>,
    AVal: Nat,
    Eq<Const<AVal>, B>: EqRightDispatch<Global, Lex, AVal, B, <B as Eval<Global, Lex>>::Value>,
{
    type Out = <Eq<Const<AVal>, B> as EqRightDispatch<
        Global,
        Lex,
        AVal,
        B,
        <B as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, A, B> Eval<Global, Lex> for Eq<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr + Eval<Global, Lex>,
    B: Expr,
    Eq<A, B>: EqLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>,
{
    type Value =
        <Eq<A, B> as EqLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>>::Out;
}

pub trait EqualpLeftDispatch<Global: AList, Lex: AList, A: Expr, B: Expr, CA: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, A: Expr, B: Expr, Tag, V>
    EqualpLeftDispatch<Global, Lex, A, B, Escape<Tag, V>> for Equalp<A, B>
{
    type Out = Escape<Tag, V>;
}

pub trait EqualpRightDispatch<Global: AList, Lex: AList, AVal: Expr, B: Expr, CB: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, AVal: Expr, B: Expr, Tag, V>
    EqualpRightDispatch<Global, Lex, AVal, B, Escape<Tag, V>> for Equalp<Const<AVal>, B>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, AVal: Expr, B: Expr, BVal>
    EqualpRightDispatch<Global, Lex, AVal, B, Ok<BVal>> for Equalp<Const<AVal>, B>
where
    AVal: EqualpVal<BVal>,
{
    type Out = Ok<<AVal as EqualpVal<BVal>>::Output>;
}

impl<Global, Lex, A, B, AVal> EqualpLeftDispatch<Global, Lex, A, B, Ok<AVal>> for Equalp<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr,
    B: Expr + Eval<Global, Lex>,
    AVal: Expr,
    Equalp<Const<AVal>, B>:
        EqualpRightDispatch<Global, Lex, AVal, B, <B as Eval<Global, Lex>>::Value>,
{
    type Out = <Equalp<Const<AVal>, B> as EqualpRightDispatch<
        Global,
        Lex,
        AVal,
        B,
        <B as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, A, B> Eval<Global, Lex> for Equalp<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr + Eval<Global, Lex>,
    B: Expr,
    Equalp<A, B>: EqualpLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>,
{
    type Value = <Equalp<A, B> as EqualpLeftDispatch<
        Global,
        Lex,
        A,
        B,
        <A as Eval<Global, Lex>>::Value,
    >>::Out;
}

pub trait LtLeftDispatch<Global: AList, Lex: AList, A: Expr, B: Expr, CA: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, A: Expr, B: Expr, Tag, V>
    LtLeftDispatch<Global, Lex, A, B, Escape<Tag, V>> for Lt<A, B>
{
    type Out = Escape<Tag, V>;
}

pub trait LtRightDispatch<Global: AList, Lex: AList, AVal: Nat, B: Expr, CB: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, Tag, V>
    LtRightDispatch<Global, Lex, AVal, B, Escape<Tag, V>> for Lt<Const<AVal>, B>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, BVal: Nat>
    LtRightDispatch<Global, Lex, AVal, B, Ok<BVal>> for Lt<Const<AVal>, B>
where
    AVal: LtNat<BVal>,
{
    type Out = Ok<<AVal as LtNat<BVal>>::Output>;
}

impl<Global, Lex, A, B, AVal> LtLeftDispatch<Global, Lex, A, B, Ok<AVal>> for Lt<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr,
    B: Expr + Eval<Global, Lex>,
    AVal: Nat,
    Lt<Const<AVal>, B>: LtRightDispatch<Global, Lex, AVal, B, <B as Eval<Global, Lex>>::Value>,
{
    type Out = <Lt<Const<AVal>, B> as LtRightDispatch<
        Global,
        Lex,
        AVal,
        B,
        <B as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, A, B> Eval<Global, Lex> for Lt<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr + Eval<Global, Lex>,
    B: Expr,
    Lt<A, B>: LtLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>,
{
    type Value =
        <Lt<A, B> as LtLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>>::Out;
}

pub trait GtLeftDispatch<Global: AList, Lex: AList, A: Expr, B: Expr, CA: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, A: Expr, B: Expr, Tag, V>
    GtLeftDispatch<Global, Lex, A, B, Escape<Tag, V>> for Gt<A, B>
{
    type Out = Escape<Tag, V>;
}

pub trait GtRightDispatch<Global: AList, Lex: AList, AVal: Nat, B: Expr, CB: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, Tag, V>
    GtRightDispatch<Global, Lex, AVal, B, Escape<Tag, V>> for Gt<Const<AVal>, B>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, AVal: Nat, B: Expr, BVal: Nat>
    GtRightDispatch<Global, Lex, AVal, B, Ok<BVal>> for Gt<Const<AVal>, B>
where
    AVal: GtNat<BVal>,
{
    type Out = Ok<<AVal as GtNat<BVal>>::Output>;
}

impl<Global, Lex, A, B, AVal> GtLeftDispatch<Global, Lex, A, B, Ok<AVal>> for Gt<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr,
    B: Expr + Eval<Global, Lex>,
    AVal: Nat,
    Gt<Const<AVal>, B>: GtRightDispatch<Global, Lex, AVal, B, <B as Eval<Global, Lex>>::Value>,
{
    type Out = <Gt<Const<AVal>, B> as GtRightDispatch<
        Global,
        Lex,
        AVal,
        B,
        <B as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, A, B> Eval<Global, Lex> for Gt<A, B>
where
    Global: AList,
    Lex: AList,
    A: Expr + Eval<Global, Lex>,
    B: Expr,
    Gt<A, B>: GtLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>,
{
    type Value =
        <Gt<A, B> as GtLeftDispatch<Global, Lex, A, B, <A as Eval<Global, Lex>>::Value>>::Out;
}

pub trait IfDispatch<Global: AList, Lex: AList, Cond: Expr, Then: Expr, Else: Expr, CCond: Control>
{
    type Out: Control;
}

impl<Global: AList, Lex: AList, Cond: Expr, Then: Expr, Else: Expr, Tag, V>
    IfDispatch<Global, Lex, Cond, Then, Else, Escape<Tag, V>> for If<Cond, Then, Else>
{
    type Out = Escape<Tag, V>;
}

impl<Global, Lex, Cond, Then, Else> Eval<Global, Lex> for If<Cond, Then, Else>
where
    Global: AList,
    Lex: AList,
    Cond: Expr + Eval<Global, Lex>,
    Then: Expr,
    Else: Expr,
    If<Cond, Then, Else>:
        IfDispatch<Global, Lex, Cond, Then, Else, <Cond as Eval<Global, Lex>>::Value>,
{
    type Value = <If<Cond, Then, Else> as IfDispatch<
        Global,
        Lex,
        Cond,
        Then,
        Else,
        <Cond as Eval<Global, Lex>>::Value,
    >>::Out;
}

pub trait IfBoolDispatch<Global: AList, Lex: AList, Then: Expr, Else: Expr, B: BoolT> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, Then: Expr, Else: Expr>
    IfBoolDispatch<Global, Lex, Then, Else, True> for True
where
    Then: Eval<Global, Lex>,
{
    type Out = <Then as Eval<Global, Lex>>::Value;
}

impl<Global: AList, Lex: AList, Then: Expr, Else: Expr>
    IfBoolDispatch<Global, Lex, Then, Else, False> for False
where
    Else: Eval<Global, Lex>,
{
    type Out = <Else as Eval<Global, Lex>>::Value;
}

impl<Global, Lex, Cond, Then, Else, B> IfDispatch<Global, Lex, Cond, Then, Else, Ok<B>>
    for If<Cond, Then, Else>
where
    Global: AList,
    Lex: AList,
    Cond: Expr,
    Then: Expr,
    Else: Expr,
    B: BoolT,
    B: IfBoolDispatch<Global, Lex, Then, Else, B>,
{
    type Out = <B as IfBoolDispatch<Global, Lex, Then, Else, B>>::Out;
}

pub trait CallFunDispatch<Global: AList, Lex: AList, F: Expr, Args: List, CF: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, F: Expr, Args: List, Tag, V>
    CallFunDispatch<Global, Lex, F, Args, Escape<Tag, V>> for Call<F, Args>
{
    type Out = Escape<Tag, V>;
}

pub trait CallArgsDispatch<Global: AList, Lex: AList, FVal, Args: List, CArgs: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, FVal, Args: List, Tag, V>
    CallArgsDispatch<Global, Lex, FVal, Args, Escape<Tag, V>> for Call<Const<FVal>, Args>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, FVal, Args: List, ArgsVal: List>
    CallArgsDispatch<Global, Lex, FVal, Args, Ok<ArgsVal>> for Call<Const<FVal>, Args>
where
    FVal: Apply<Global, ArgsVal>,
{
    type Out = <FVal as Apply<Global, ArgsVal>>::Output;
}

impl<Global, Lex, F, Args, FVal> CallFunDispatch<Global, Lex, F, Args, Ok<FVal>> for Call<F, Args>
where
    Global: AList,
    Lex: AList,
    F: Expr,
    Args: List + EvalList<Global, Lex>,
    Call<Const<FVal>, Args>:
        CallArgsDispatch<Global, Lex, FVal, Args, <Args as EvalList<Global, Lex>>::Output>,
{
    type Out = <Call<Const<FVal>, Args> as CallArgsDispatch<
        Global,
        Lex,
        FVal,
        Args,
        <Args as EvalList<Global, Lex>>::Output,
    >>::Out;
}

impl<Global, Lex, F, Args> Eval<Global, Lex> for Call<F, Args>
where
    Global: AList,
    Lex: AList,
    F: Expr + Eval<Global, Lex>,
    Args: List,
    Call<F, Args>: CallFunDispatch<Global, Lex, F, Args, <F as Eval<Global, Lex>>::Value>,
{
    type Value = <Call<F, Args> as CallFunDispatch<
        Global,
        Lex,
        F,
        Args,
        <F as Eval<Global, Lex>>::Value,
    >>::Out;
}

pub trait ApplyFunDispatch<Global: AList, Lex: AList, F: Expr, ArgsList: Expr, CF: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, F: Expr, ArgsList: Expr, Tag, V>
    ApplyFunDispatch<Global, Lex, F, ArgsList, Escape<Tag, V>> for ApplyExpr<F, ArgsList>
{
    type Out = Escape<Tag, V>;
}

pub trait ApplyArgsDispatch<Global: AList, Lex: AList, FVal, ArgsList: Expr, CArgs: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, FVal, ArgsList: Expr, Tag, V>
    ApplyArgsDispatch<Global, Lex, FVal, ArgsList, Escape<Tag, V>>
    for ApplyExpr<Const<FVal>, ArgsList>
{
    type Out = Escape<Tag, V>;
}

impl<Global: AList, Lex: AList, FVal, ArgsList: Expr, ArgsVal: List>
    ApplyArgsDispatch<Global, Lex, FVal, ArgsList, Ok<ArgsVal>> for ApplyExpr<Const<FVal>, ArgsList>
where
    FVal: Apply<Global, ArgsVal>,
{
    type Out = <FVal as Apply<Global, ArgsVal>>::Output;
}

impl<Global, Lex, F, ArgsList, FVal> ApplyFunDispatch<Global, Lex, F, ArgsList, Ok<FVal>>
    for ApplyExpr<F, ArgsList>
where
    Global: AList,
    Lex: AList,
    F: Expr,
    ArgsList: Expr + Eval<Global, Lex>,
    ApplyExpr<Const<FVal>, ArgsList>:
        ApplyArgsDispatch<Global, Lex, FVal, ArgsList, <ArgsList as Eval<Global, Lex>>::Value>,
{
    type Out = <ApplyExpr<Const<FVal>, ArgsList> as ApplyArgsDispatch<
        Global,
        Lex,
        FVal,
        ArgsList,
        <ArgsList as Eval<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, F, ArgsList> Eval<Global, Lex> for ApplyExpr<F, ArgsList>
where
    Global: AList,
    Lex: AList,
    F: Expr + Eval<Global, Lex>,
    ArgsList: Expr,
    ApplyExpr<F, ArgsList>:
        ApplyFunDispatch<Global, Lex, F, ArgsList, <F as Eval<Global, Lex>>::Value>,
{
    type Value = <ApplyExpr<F, ArgsList> as ApplyFunDispatch<
        Global,
        Lex,
        F,
        ArgsList,
        <F as Eval<Global, Lex>>::Value,
    >>::Out;
}

/// `call/ec` (escape-continuation flavor, explicit tag).
///
/// This is *delimited* to the dynamic extent of evaluating the function `F`.
///
/// Why the explicit `Tag`?
/// - In a type-level interpreter on stable Rust we need a way to distinguish
///   nested continuations without specialization.
/// - Requiring `Tag: Key` gives us a cheap, type-level equality test via
///   `EqKey`.
#[derive(Debug, Clone, Copy, Default)]
pub struct CallEC<Tag: Key, F: Expr>(PhantomData<(Tag, F)>);
impl<Tag: Key, F: Expr> Expr for CallEC<Tag, F> {}

/// A continuation value passed to the function in `call/ec`.
///
/// Applying it escapes back to the `CallEC` with the same `Tag`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cont<Tag: Key>(PhantomData<Tag>);

impl<Tag: Key, Global: AList, A> Apply<Global, Cons<A, Nil>> for Cont<Tag> {
    type Output = Escape<Tag, A>;
}

pub trait CatchEscape<Expected: Key>: Control {
    type Out: Control;
}

impl<Expected: Key, V> CatchEscape<Expected> for Ok<V> {
    type Out = Ok<V>;
}

pub trait CatchEscapeDispatch<Expected: Key, Match: BoolT>: Control {
    type Out: Control;
}

impl<Expected: Key, Tag: Key, V> CatchEscapeDispatch<Expected, True> for Escape<Tag, V> {
    type Out = Ok<V>;
}

impl<Expected: Key, Tag: Key, V> CatchEscapeDispatch<Expected, False> for Escape<Tag, V> {
    type Out = Escape<Tag, V>;
}

impl<Expected: Key, Tag: Key, V> CatchEscape<Expected> for Escape<Tag, V>
where
    Tag: EqKey<Expected>,
    Escape<Tag, V>: CatchEscapeDispatch<Expected, <Tag as EqKey<Expected>>::Output>,
{
    type Out =
        <Escape<Tag, V> as CatchEscapeDispatch<Expected, <Tag as EqKey<Expected>>::Output>>::Out;
}

pub trait CallECDispatch<Global: AList, Lex: AList, Tag: Key, F: Expr, CF: Control> {
    type Out: Control;
}

impl<Global: AList, Lex: AList, Tag: Key, F: Expr, EscTag, V>
    CallECDispatch<Global, Lex, Tag, F, Escape<EscTag, V>> for CallEC<Tag, F>
{
    type Out = Escape<EscTag, V>;
}

impl<Global, Lex, Tag, F, FVal> CallECDispatch<Global, Lex, Tag, F, Ok<FVal>> for CallEC<Tag, F>
where
    Global: AList,
    Lex: AList,
    Tag: Key,
    F: Expr,
    FVal: Apply<Global, Cons<Cont<Tag>, Nil>>,
    <FVal as Apply<Global, Cons<Cont<Tag>, Nil>>>::Output: CatchEscape<Tag>,
{
    type Out = <<FVal as Apply<Global, Cons<Cont<Tag>, Nil>>>::Output as CatchEscape<Tag>>::Out;
}

impl<Global, Lex, Tag, F> Eval<Global, Lex> for CallEC<Tag, F>
where
    Global: AList,
    Lex: AList,
    Tag: Key,
    F: Expr + Eval<Global, Lex>,
    CallEC<Tag, F>: CallECDispatch<Global, Lex, Tag, F, <F as Eval<Global, Lex>>::Value>,
{
    type Value = <CallEC<Tag, F> as CallECDispatch<
        Global,
        Lex,
        Tag,
        F,
        <F as Eval<Global, Lex>>::Value,
    >>::Out;
}

/// Evaluate a top-level form (e.g. `Defun`) and produce a new environment.
pub trait EvalForm<Global: AList, Lex: AList>: Form {
    type GlobalOut: AList;
    type Value;
}

impl<Global, Lex, Name, Params, Body> EvalForm<Global, Lex> for Defun<Name, Params, Body>
where
    Global: AList,
    Lex: AList,
    Name: Key,
    Params: List,
    Body: Expr,
{
    type GlobalOut = ACons<Name, RecClosure<Name, Global, Lex, Params, Body>, Global>;
    type Value = Name;
}

/// Evaluate a single `begin` item.
///
/// This is either a normal expression (which does not change the global env),
/// or a top-level form like `defun` (which produces an updated global env).
pub trait EvalBeginItem<Global: AList, Lex: AList> {
    type GlobalOut: AList;
    type Value: Control;
}

impl<Global, Lex, E> EvalBeginItem<Global, Lex> for E
where
    Global: AList,
    Lex: AList,
    E: Expr + Eval<Global, Lex>,
{
    type GlobalOut = Global;
    type Value = <E as Eval<Global, Lex>>::Value;
}

impl<Global, Lex, Name, Params, Body> EvalBeginItem<Global, Lex> for Defun<Name, Params, Body>
where
    Global: AList,
    Lex: AList,
    Name: Key,
    Params: List,
    Body: Expr,
{
    type GlobalOut = <Defun<Name, Params, Body> as EvalForm<Global, Lex>>::GlobalOut;
    type Value = Ok<<Defun<Name, Params, Body> as EvalForm<Global, Lex>>::Value>;
}

pub trait EvalBegin<Global: AList, Lex: AList>: List {
    type GlobalOut: AList;
    type Output: Control;
}

impl<Global: AList, Lex: AList> EvalBegin<Global, Lex> for Nil {
    type GlobalOut = Global;
    type Output = Ok<Nil>;
}

impl<Global, Lex, H> EvalBegin<Global, Lex> for Cons<H, Nil>
where
    Global: AList,
    Lex: AList,
    H: EvalBeginItem<Global, Lex>,
{
    type GlobalOut = <H as EvalBeginItem<Global, Lex>>::GlobalOut;
    type Output = <H as EvalBeginItem<Global, Lex>>::Value;
}

pub trait EvalBeginDispatch<Global: AList, Lex: AList, HGlobal: AList, T: List, CH: Control> {
    type GlobalOut: AList;
    type Out: Control;
}

impl<Global: AList, Lex: AList, HGlobal: AList, H, T: List, Tag, V>
    EvalBeginDispatch<Global, Lex, HGlobal, T, Escape<Tag, V>> for Cons<H, T>
{
    type GlobalOut = HGlobal;
    type Out = Escape<Tag, V>;
}

impl<Global, Lex, HGlobal, H, T, HVal> EvalBeginDispatch<Global, Lex, HGlobal, T, Ok<HVal>>
    for Cons<H, T>
where
    Global: AList,
    Lex: AList,
    HGlobal: AList,
    T: List + EvalBegin<HGlobal, Lex>,
{
    type GlobalOut = <T as EvalBegin<HGlobal, Lex>>::GlobalOut;
    type Out = <T as EvalBegin<HGlobal, Lex>>::Output;
}

impl<Global, Lex, H, T> EvalBegin<Global, Lex> for Cons<H, T>
where
    Global: AList,
    Lex: AList,
    H: EvalBeginItem<Global, Lex>,
    T: NonNilList,
    Cons<H, T>: EvalBeginDispatch<
            Global,
            Lex,
            <H as EvalBeginItem<Global, Lex>>::GlobalOut,
            T,
            <H as EvalBeginItem<Global, Lex>>::Value,
        >,
{
    type GlobalOut = <Cons<H, T> as EvalBeginDispatch<
        Global,
        Lex,
        <H as EvalBeginItem<Global, Lex>>::GlobalOut,
        T,
        <H as EvalBeginItem<Global, Lex>>::Value,
    >>::GlobalOut;

    type Output = <Cons<H, T> as EvalBeginDispatch<
        Global,
        Lex,
        <H as EvalBeginItem<Global, Lex>>::GlobalOut,
        T,
        <H as EvalBeginItem<Global, Lex>>::Value,
    >>::Out;
}

impl<Global, Lex, Seq> Eval<Global, Lex> for Begin<Seq>
where
    Global: AList,
    Lex: AList,
    Seq: List + EvalBegin<Global, Lex>,
{
    type Value = <Seq as EvalBegin<Global, Lex>>::Output;
}

/// Helper trait for retrieving the arity (parameter count) of lambdas/defuns.
pub trait Arity {
    const VALUE: usize;
}

impl<Params, Body> Arity for Lambda<Params, Body>
where
    Params: Length + List,
{
    const VALUE: usize = <Params as Length>::VALUE;
}

impl<Name, Params, Body> Arity for Defun<Name, Params, Body>
where
    Params: Length + List,
{
    const VALUE: usize = <Params as Length>::VALUE;
}

const fn assert_usize_eq(left: usize, right: usize) {
    assert!(left == right);
}

// Generated by build.rs (expr_nat! macro for 0..16384)
include!(concat!(env!("OUT_DIR"), "/generated_expr_nat.rs"));

macro_rules! expr {
    ((if $test: tt $then: tt $else: tt)) => {
        $crate::If<expr!($test), expr!($then),  expr!($else)>
    };

    ((cons $a:tt $b:tt)) => {
        $crate::Cons<expr!($a), expr!($b)>
    };
    (nil) => {
        $crate::Nil
    };

    ((lambda ( $( $param:ident )* ) $body:tt)) => {
        $crate::Lambda<list_t!($( $param ),*), expr!($body)>
    };

    ((begin $( $e:tt )+)) => {
        $crate::Begin<list_t!($(expr!($e)),*)>
    };

    // Support both `(call/ec Tag f)` and `(callec Tag f)` spellings.
    ((call / ec $tag:ident $f:tt)) => {
        $crate::CallEC<$tag, expr!($f)>
    };

    ((callec $tag:ident $f:tt)) => {
        $crate::CallEC<$tag, expr!($f)>
    };

    ((cdr $a:tt)) => {
        $crate::Cdr<expr!($a)>
    };

    ((car $a:tt)) => {
        $crate::Car<expr!($a)>
    };

    ((equalp $a:tt $b:tt)) => {
        $crate::Equalp<expr!($a), expr!($b)>
    };

    ((let ( $( ( $k:ident $v:tt ) )* ) $body:tt)) => {
        let_!(
            ( $( $k = expr!($v) ),* ),
            expr!($body)
        )
    };

    ((defun $name:ident ( $( $param:ident )* ) $body:tt)) => {
        defun!(
            $name,
            ( $( $param ),* ),
            expr!($body)
        )
    };

    (($name : ident $( $arg:tt )* )) => {
        call!(
            $crate::Var<$name>,
            $(expr!($arg)),*
        )
    };

    ($k:ident) => {
        $crate::Var<$k>
    };

    ((+ $a:tt $b:tt)) => {
        $crate::Plus<expr!($a), expr!($b)>
    };

    ((* $a:tt $b:tt)) => {
        $crate::Mul<expr!($a), expr!($b)>
    };

    ((- $a:tt $b:tt)) => {
        $crate::Sub<expr!($a), expr!($b)>
    };

    ((= $a:tt $b:tt)) => {
        $crate::Eq<expr!($a), expr!($b)>
    };

    ((equalp $a:tt $b:tt)) => {
        $crate::Equalp<expr!($a), expr!($b)>
    };

    ((< $a:tt $b:tt)) => {
        $crate::Lt<expr!($a), expr!($b)>
    };

    ((> $a:tt $b:tt)) => {
        $crate::Gt<expr!($a), expr!($b)>
    };

    ($k: ident = $v:tt) => {
        $crate::SetBang<$k, expr!($v), expr!(body)>
    };

    ($n:tt) => {
        expr_nat!($n)
    };






}

fn main() {
    // Example: (1 2 3) encoded at the type-level.
    struct One;
    struct Two;
    struct Three;

    type L123 = Cons<One, Cons<Two, Cons<Three, Nil>>>;

    // Compile-time check that the list length is 3.
    const _: () = assert_usize_eq(<L123 as Length>::VALUE, 3);

    println!(
        "Type-level list length computed: {}",
        <L123 as Length>::VALUE
    );

    // Lambda/defun examples.
    struct X;
    struct Y;
    struct Add;
    struct PlusBody;

    type Params2 = Cons<X, Cons<Y, Nil>>;
    type AddLambda = lambda!((X, Y), PlusBody);
    type AddDefun = Defun<Add, Params2, PlusBody>;

    const _: () = assert_usize_eq(<AddLambda as Arity>::VALUE, 2);
    const _: () = assert_usize_eq(<AddDefun as Arity>::VALUE, 2);

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

defkey!(SymLength, N0);

type LengthFunc = expr!(
    (defun SymLength (SymLst)
        (if (equalp SymLst nil)
            0
            (+ 1 (SymLength (cdr SymLst))))));

type GlobalLength = <LengthFunc as EvalForm<ANil, ANil>>::GlobalOut;
type LengthOfList =
    EvalValue<expr!((SymLength (cons 1 (cons 2 (cons 3 nil))))), GlobalLength, ANil>;
const _: () = assert_same::<LengthOfList, N3>();

defkey!(SymMap, N0);
defkey!(SymFunc, N1);
defkey!(SymLst, N2);
defkey!(SymTmp, N3);
defkey!(SymOne, N4);

type MapFunc = expr!(
    (defun SymMap (SymFunc SymLst)
        (if (equalp SymLst nil)
            nil
            (cons
                (SymFunc (car SymLst))
                (SymMap SymFunc (cdr SymLst))))));

type GlobalMap = <MapFunc as EvalForm<ANil, ANil>>::GlobalOut;
type MapResult = EvalValue<
    expr!(
    (let ((SymOne 1))
        (SymMap (lambda(SymTmp) (+ SymOne SymTmp)) (cons 1 (cons 2 (cons 3 nil)))))),
    GlobalMap,
    ANil,
>;

const _: () = assert_same::<MapResult, Cons<N2, Cons<N3, Cons<N4, Nil>>>>();
