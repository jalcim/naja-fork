cmake -S. -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/tmp/naja/ && cmake --build build && cmake --build build --target test
