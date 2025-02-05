use criterion::{criterion_group, criterion_main, Criterion};

use linear_algebra_ml::dense::matrix::Matrix;
use linear_algebra_ml::dense::ops;

fn bench_mul(c: &mut Criterion) {
    let a = Matrix::new(100, 100, vec![1.0; 100 * 100]);
    let b = Matrix::new(100, 100, vec![2.0; 100 * 100]);

    c.bench_function("matrix multiplication 100x100", |bencher| {
        bencher.iter(|| ops::mul(&a, &b).unwrap())
    });
}

criterion_group!(benches, bench_mul);
criterion_main!(benches);
