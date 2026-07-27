@echo off
rem Launch the local motion server (viewer at http://localhost:8888)
rem Checkpoint path is resolved from MDM_REPO by the server itself — no argument needed.
set MDM_REPO=D:\Documents\11Projects\Kimodo\local\motion-diffusion-model
cd /d D:\Documents\11Projects\Kimodo\local
venv\Scripts\python.exe -W ignore app\motion_server.py > server.log 2>&1
