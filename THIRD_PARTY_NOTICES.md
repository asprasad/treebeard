# Third-Party Notices

Treebeard binary distributions include or dynamically link components from
the projects below. These components remain subject to their respective
licenses. The exact set of transitively bundled libraries can vary by wheel;
`auditwheel show <wheel>` reports it for a particular artifact.

| Component | License | Project |
| --- | --- | --- |
| LLVM and OpenMP | Apache-2.0 WITH LLVM-exception | https://llvm.org/ |
| Protocol Buffers | BSD-3-Clause | https://protobuf.dev/ |
| libedit | BSD-3-Clause | https://www.thrysoee.dk/editline/ |
| libbsd and libmd | BSD-family licenses | https://libbsd.freedesktop.org/ |
| ICU | Unicode-3.0 | https://icu.unicode.org/ |
| GNU libiconv | LGPL-2.1-or-later | https://www.gnu.org/software/libiconv/ |
| libffi | MIT | https://sourceware.org/libffi/ |
| libxml2 | MIT | https://gitlab.gnome.org/GNOME/libxml2 |
| ncurses | X11-style | https://invisible-island.net/ncurses/ |
| XZ Utils | 0BSD and public-domain components | https://tukaani.org/xz/ |
| Zstandard | BSD-3-Clause | https://facebook.github.io/zstd/ |

Treebeard itself is distributed under the MIT license in `LICENSE`.