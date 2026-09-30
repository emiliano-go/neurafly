//! Release-consistency check.
//!
//! Normal `cargo test` runs pass no `--prerelease`, so this is a no-op there.
//! Release workflows call:
//!
//!     cargo test --test release -- --prerelease
//!
//! to fail the release if the crate version and the AUR package were not
//! bumped and re-pinned together.

use std::path::Path;

fn main() {
    if !std::env::args().any(|a| a == "--prerelease") {
        return;
    }

    let version = env!("CARGO_PKG_VERSION");
    let pkgbuild = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("packaging/aur/PKGBUILD"),
    )
    .expect("packaging/aur/PKGBUILD is missing");

    let pkgver = pkgbuild
        .lines()
        .find_map(|l| l.trim().strip_prefix("pkgver="))
        .expect("no pkgver= line in PKGBUILD");
    assert_eq!(
        version,
        pkgver.trim(),
        "Cargo.toml version ({version}) and AUR pkgver ({}) differ",
        pkgver.trim()
    );

    let sums = pkgbuild
        .lines()
        .find_map(|l| l.trim().strip_prefix("sha256sums="))
        .expect("no sha256sums= line in PKGBUILD");
    let hash = sums.trim().trim_start_matches("('").trim_end_matches("')");
    assert!(
        hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()),
        "AUR sha256sums is not a pinned 64-hex digest: {sums}"
    );

    // On a tag build, the git tag must be v<version>.
    if std::env::var("GITHUB_REF_TYPE").as_deref() == Ok("tag") {
        let tag = std::env::var("GITHUB_REF_NAME").unwrap_or_default();
        assert_eq!(
            tag,
            format!("v{version}"),
            "git tag ({tag}) and Cargo version ({version}) differ"
        );
    }

    println!("release consistency ok: neurafly {version}");
}
