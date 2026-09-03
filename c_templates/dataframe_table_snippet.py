import pandas as pd
from infotheory import get_pipeline_paths
from infotheory.plotting import save_table_markdown, save_table_image

paths = get_pipeline_paths(__file__)
df = pd.DataFrame({"area": ["V1", "PM"], "d_prime": [2.3, 1.9]})

save_table_markdown(df, paths.out, name="dprime_by_area")
save_table_image(
    df, paths.out, name="dprime_by_area_styled",
    styler_fn=lambda d: d.style.background_gradient(subset=["d_prime"]),
)
