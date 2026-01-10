@echo off
echo Killing Flint processes...
taskkill /F /IM node.exe
taskkill /F /IM python.exe
taskkill /F /IM electron.exe
echo Done.

