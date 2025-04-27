@echo off
reg delete "HKEY_CLASSES_ROOT\lnkfile" /v IsShortcut /f
reg delete "HKEY_CLASSES_ROOT\piffile" /v IsShortcut /f
taskkill /f /im explorer.exe
start explorer.exe
