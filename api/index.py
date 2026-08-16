"""Vercel serverless entrypoint for the shopper panel app."""

from simulator.app import app

# Vercel routes all traffic to this ASGI app.
# Live Browser Use panels need a worker machine; set PANEL_WORKER_URL to proxy runs.
