"""Limpieza de texto compartida entre el cuaderno de entrenamiento y app.py.

IMPORTANTE: este archivo debe estar junto a app.py, porque el modelo .pkl
hace referencia a preprocesamiento.limpiar_texto.
"""
import re

RE_SUBJECT = re.compile(r'^\s*subject\s*:\s*', re.IGNORECASE)
RE_URL = re.compile(r'(https?\s*:\s*/\s*/\s*\S+|www\s*\.\s*\S+)', re.IGNORECASE)
RE_EMAIL = re.compile(r'\b[\w.+-]+\s*@\s*[\w-]+(?:\s*\.\s*[\w-]+)+\b')
RE_DINERO = re.compile(r'(å£|£|\$|€|¥)')
RE_TELEFONO = re.compile(r'(?<!\d)\+?\d[\d\s-]{5,}\d(?!\d)')
RE_NUMERO = re.compile(r'\d+')
RE_ESPACIOS = re.compile(r'\s+')


def limpiar_texto(texto):
    """Normaliza un mensaje (SMS o correo) antes de vectorizarlo."""
    if texto is None:
        return ''
    t = str(texto)
    t = RE_SUBJECT.sub(' ', t)
    t = t.lower()
    t = RE_URL.sub(' tokenurl ', t)
    t = RE_EMAIL.sub(' tokenemail ', t)
    t = RE_DINERO.sub(' tokendinero ', t)
    t = RE_TELEFONO.sub(' tokentelefono ', t)
    t = RE_NUMERO.sub(' tokennumero ', t)
    return RE_ESPACIOS.sub(' ', t).strip()
