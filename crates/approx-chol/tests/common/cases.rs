/// `#[rstest]` whose `arg in list(subset)` parameters take their values from the lists below.
macro_rules! cases {
    ($(#[$meta:meta])* fn $name:ident($($params:tt)*) $body:block) => {
        $crate::cases::cases!(@param [$(#[$meta])* fn $name] [] [$($params)*] $body);
    };
    (@param $head:tt $done:tt [$arg:ident in $list:ident $(($subset:ident))? $(, $($rest:tt)*)?]
        $body:block) => {
        $crate::cases::cases!(@list $list [$($subset)?] [$arg $head $done [$($($rest)*)?] $body]);
    };
    (@param $head:tt [$($done:tt)*] [#[$attr:meta] $arg:ident: $ty:ty $(, $($rest:tt)*)?]
        $body:block) => {
        $crate::cases::cases!(@param $head [$($done)* #[$attr] $arg: $ty,] [$($($rest)*)?] $body);
    };
    (@param [$(#[$meta:meta])* fn $name:ident] [$($done:tt)*] [] $body:block) => {
        #[rstest::rstest]
        $(#[$meta])*
        fn $name($($done)*) $body
    };
    (@values $ty:tt [$($values:tt)*] [$arg:ident $head:tt [$($done:tt)*] $rest:tt $body:block]) => {
        $crate::cases::cases!(@param $head [$($done)* #[values($($values)*)] $arg: $ty,]
            $rest $body);
    };

    (@list entries [$subset:ident] $k:tt) => {
        $crate::cases::cases!(@entries $subset [@named Entry $k]);
    };
    (@list types [$subset:ident] $k:tt) => {
        $crate::cases::cases!(@keep $subset [] [F32U32 F32U64 F64U32 F64U64] [@named Types $k]);
    };
    (@list routes [$subset:ident] $k:tt) => {
        $crate::cases::cases!(@entries $subset [@routes modes [] $k]);
    };
    (@list sparse_routes [$subset:ident] $k:tt) => {
        $crate::cases::cases!(@entries $subset [@routes sparse [] $k]);
    };
    (@entries $subset:ident $then:tt) => {
        $crate::cases::cases!(@keep $subset [] [
            Csr UpperCsr FactorWidth2 BipartiteFactorWidth2 Laplacian GroundedLaplacian
        ] $then);
    };
    (@named [$($member:ident)*] $ty:ident $k:tt) => {
        $crate::cases::cases!(@values $ty [$($ty::$member),*] $k);
    };

    // Only a bipartite block reads the complement rule, and only a bipartite factor-width-2 Gram builds one.
    (@routes [BipartiteFactorWidth2 $($rest:ident)*] $kind:ident $acc:tt $k:tt) => {
        $crate::cases::cases!(@sparse [@route $kind BipartiteFactorWidth2 [$($rest)*] $acc $k]);
    };
    (@routes [$entry:ident $($rest:ident)*] $kind:ident $acc:tt $k:tt) => {
        $crate::cases::cases!(@route $kind $entry [$($rest)*] $acc $k
            [AcExactComplement Ac2ExactComplement]);
    };
    (@routes [] modes [$($acc:tt)*] $k:tt) => {
        $crate::cases::cases!(@values (Entry, Mode) [$($acc)*] $k);
    };
    (@routes [] sparse [$($acc:tt)*] $k:tt) => {
        $crate::cases::cases!(@values (Entry, Sparse) [$($acc)*] $k);
    };
    (@route modes $entry:ident $rest:tt [$($acc:tt)*] $k:tt [$($sparse:ident)*]) => {
        $crate::cases::cases!(@routes $rest modes [$($acc)*
            (Entry::$entry, Mode::Dense),
            $((Entry::$entry, Mode::Sparse(Sparse::$sparse)),)*
        ] $k);
    };
    (@route sparse $entry:ident $rest:tt [$($acc:tt)*] $k:tt [$($sparse:ident)*]) => {
        $crate::cases::cases!(@routes $rest sparse [$($acc)*
            $((Entry::$entry, Sparse::$sparse),)*
        ] $k);
    };
    (@sparse [@$step:ident $($then:tt)*]) => {
        $crate::cases::cases!(@$step $($then)* [
            AcExactComplement AcComplementAc AcComplementAc2
            Ac2ExactComplement Ac2ComplementAc Ac2ComplementAc2
        ]);
    };

    (@keep $subset:ident [$($kept:ident)*] [$member:ident $($rest:ident)*] $then:tt) => {
        $crate::cases::cases!(@excludes $subset $member
            [@keep $subset [$($kept)*] [$($rest)*] $then]
            [@keep $subset [$($kept)* $member] [$($rest)*] $then]);
    };
    (@keep $subset:ident $kept:tt [] [@$step:ident $($then:tt)*]) => {
        $crate::cases::cases!(@$step $kept $($then)*);
    };
    // A subset leaves out the entry points that cannot state its graphs, or the other scalar.
    (@excludes grounded Laplacian $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes frustrated BipartiteFactorWidth2 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes frustrated Laplacian $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes frustrated GroundedLaplacian $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes lone_ground FactorWidth2 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes lone_ground BipartiteFactorWidth2 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes lone_ground Laplacian $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes non_bipartite BipartiteFactorWidth2 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes f32 F64U32 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes f32 F64U64 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes f64 F32U32 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes f64 F32U64 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes $subset:ident $member:ident $out:tt $in:tt) => { $crate::cases::cases! $in; };
}
pub(crate) use cases;
