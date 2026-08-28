"""Shared visual tokens for the light desktop research interface."""

import customtkinter as ctk


BACKGROUND = "#F5F7FB"
SURFACE = "#FFFFFF"
SURFACE_MUTED = "#F8FAFD"
BORDER = "#DCE3ED"
TEXT = "#162033"
MUTED = "#64748B"
PRIMARY = "#1769E0"
PRIMARY_HOVER = "#1257BD"
SOFT_BLUE = "#EAF2FF"
SUCCESS = "#15803D"
SOFT_GREEN = "#EAF7EE"
DANGER = "#B42318"
WARNING = "#B45309"


def card(parent, **kwargs):
    options = {
        "fg_color": SURFACE,
        "border_color": BORDER,
        "border_width": 1,
        "corner_radius": 10,
    }
    options.update(kwargs)
    return ctk.CTkFrame(parent, **options)


def section_title(parent, text: str):
    return ctk.CTkLabel(
        parent,
        text=text,
        font=("Segoe UI", 15, "bold"),
        text_color=TEXT,
        anchor="w",
    )


def muted_label(parent, text: str, **kwargs):
    return ctk.CTkLabel(
        parent,
        text=text,
        font=("Segoe UI", 12),
        text_color=MUTED,
        **kwargs,
    )
