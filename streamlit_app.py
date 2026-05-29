"""
Punto de entrada para Streamlit Cloud.
Streamlit Cloud ejecuta este archivo desde la raíz del repo,
por lo que necesitamos agregar src/ al PYTHONPATH manualmente
antes de importar cualquier módulo del proyecto.
"""
import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.join(_root, "src")

for _p in [_src, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.main import main
main()
