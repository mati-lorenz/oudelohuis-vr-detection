# -*- coding: utf-8 -*-
"""
data/loaders.py
===============
YOUR lab-specific raw-data reading lives here, and ONLY here. Implement data.session.SessionLoader against your actual file format(s); nothing downstream should need to know what that format is.

TODO(you):
- class YourLabSessionLoader(SessionLoader): list_sessions(protocol), load(session_id, **kwargs) -> NeuralSession
- Keep loading lazy where possible (e.g. calciumdata) if sessions are large -- decide a convention (attribute set to None until requested vs. always eager) and stick to it everywhere.
- If you need a performance/inclusion filter (e.g. behavioral performance threshold for a detection task), implement it as a filter over the output of list_sessions/load, not baked into the loader itself, so it stays swappable.
"""

import numpy as np  # TODO: trim/add imports as needed

