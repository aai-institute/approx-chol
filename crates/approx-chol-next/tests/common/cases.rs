/// `#[rstest]` whose `arg in list(subset)` parameters take their values from the lists below.
macro_rules! cases {
    ($(#[$meta:meta])* fn $name:ident($($params:tt)*) $body:block) => {
        $crate::cases::cases!(@param [$(#[$meta])* fn $name] [] [$($params)*] $body);
    };
    (@param $head:tt $done:tt [$arg:ident in $list:ident $(($subset:ident))? $(, $($rest:tt)*)?] $body:block) => {
        $crate::cases::cases!(@list $list [$($subset)?] [$arg $head $done [$($($rest)*)?] $body]);
    };
    (@param $head:tt [$($done:tt)*] [#[$attr:meta] $arg:ident: $ty:ty $(, $($rest:tt)*)?] $body:block) => {
        $crate::cases::cases!(@param $head [$($done)* #[$attr] $arg: $ty,] [$($($rest)*)?] $body);
    };
    (@param [$(#[$meta:meta])* fn $name:ident] [$($done:tt)*] [] $body:block) => {
        #[rstest::rstest]
        $(#[$meta])*
        fn $name($($done)*) $body
    };
    (@values $ty:tt [$($values:tt)*] [$arg:ident $head:tt [$($done:tt)*] $rest:tt $body:block]) => {
        $crate::cases::cases!(@param $head [$($done)* #[values($($values)*)] $arg: $ty,] $rest $body);
    };

    (@list entries [$subset:ident] $k:tt) => {
        $crate::cases::cases!(@keep $subset [] [
            Csr UpperCsr Gram BipartiteGram Laplacian GroundedLaplacian SignedGraph
        ] [@named Entry $k]);
    };
    (@list types [$subset:ident] $k:tt) => {
        $crate::cases::cases!(@keep $subset [] [F32U32 F32U64 F64U32 F64U64] [@named Types $k]);
    };
    (@list sparse [] $k:tt) => {
        $crate::cases::cases!(@sparse [@named Sparse $k]);
    };
    (@list modes [] $k:tt) => {
        $crate::cases::cases!(@sparse [@modes $k]);
    };
    (@sparse [@$step:ident $($then:tt)*]) => {
        $crate::cases::cases!(@$step [
            AcExactComplement AcComplementAc AcComplementAc2
            Ac2ExactComplement Ac2ComplementAc Ac2ComplementAc2
        ] $($then)*);
    };
    (@named [$($member:ident)*] $ty:ident $k:tt) => {
        $crate::cases::cases!(@values $ty [$($ty::$member),*] $k);
    };
    (@modes [$($sparse:ident)*] $k:tt) => {
        $crate::cases::cases!(@values Mode [Mode::Dense, $(Mode::Sparse(Sparse::$sparse)),*] $k);
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
    (@excludes grounded SignedGraph $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes frustrated BipartiteGram $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes frustrated Laplacian $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes frustrated GroundedLaplacian $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes lone_ground Gram $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes lone_ground BipartiteGram $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes lone_ground Laplacian $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes lone_ground SignedGraph $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes non_bipartite BipartiteGram $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes f32 F64U32 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes f32 F64U64 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes f64 F32U32 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes f64 F32U64 $out:tt $in:tt) => { $crate::cases::cases! $out; };
    (@excludes $subset:ident $member:ident $out:tt $in:tt) => { $crate::cases::cases! $in; };
}
pub(crate) use cases;
