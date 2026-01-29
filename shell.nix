{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  name = "hnefatafl-dev-shell";

  buildInputs = with pkgs; [
    rustup
    clang
    gcc
    pkg-config
    openssl
    alsa-lib
    xorg.libXcursor
    xorg.libXi
    xorg.libX11
    xorg.libXrandr
    xorg.libxcb
    xorg.libXinerama
    xorg.libXext
    xorg.libXrender
    xorg.libXxf86vm
    libxkbcommon
    mold
    libclang
    glibc
    glibc.dev
  ];

  shellHook = ''
    export LIBCLANG_PATH=${pkgs.libclang.lib}/lib
    export RUSTUP_HOME=$PWD/.rustup
    export CARGO_HOME=$PWD/.cargo
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.alsa-lib
      pkgs.openssl
      pkgs.glibc
      pkgs.xorg.libXcursor
      pkgs.xorg.libXi
      pkgs.xorg.libX11
      pkgs.xorg.libXrandr
      pkgs.xorg.libxcb
      pkgs.xorg.libXinerama
      pkgs.xorg.libXext
      pkgs.xorg.libXrender
      pkgs.xorg.libXxf86vm
      pkgs.libxkbcommon
    ]}:$LD_LIBRARY_PATH
    if ! rustup show active-toolchain > /dev/null 2>&1; then
      rustup toolchain install stable >/dev/null
      rustup default stable >/dev/null
    fi
  '';
}
