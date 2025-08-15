import os, re
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
def load_csv_robusto(path):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    # último intento sin encoding explícito
    return pd.read_csv(path, low_memory=False)

# --- Parsear fechas m/d/yy (pivot: <30 -> 2000s; >=30 -> 1900s) ---
def parse_mdY(s, pivot=30):
    if pd.isna(s): 
        return pd.NaT
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s*$", str(s))
    if not m:
        return pd.NaT
    mm, dd, yy = map(int, m.groups())
    if yy < 100:
        yy = 2000 + yy if yy < pivot else 1900 + yy
    try:
        return pd.Timestamp(year=yy, month=mm, day=dd)
    except Exception:
        return pd.NaT
    
def safe_int_cast(series: pd.Series, fill_value: int | None = None) -> pd.Series:
    """
    Convierte a int de forma segura:
      - Si hay NaN y se provee fill_value, primero rellena con fill_value.
      - Si aún quedan nulos, usa Int64 y sólo al final rellena 0 y castea a int.
    """
    s = pd.to_numeric(series, errors="coerce")
    if fill_value is not None:
        s = s.fillna(fill_value)
    if s.isna().any():
        s = s.astype("Int64")
        if not s.isna().any():
            return s.astype(int)
        s = s.fillna(0).astype(int)
        return s
    return s.astype(int)

def first_mode(series: pd.Series):
    """Devuelve la primera moda de la serie (o NaN si no hay)."""
    try:
        m = series.mode(dropna=True)
        if len(m) > 0:
            return m.iloc[0]
    except Exception:
        pass
    return np.nan

def group_mode_transform(df: pd.DataFrame, by_cols: list[str], target_col: str) -> pd.Series:
    """Transform robusto de moda por grupo (toma la primera moda)."""
    return df.groupby(by_cols)[target_col].transform(lambda s: first_mode(s))

# -------------------------------------------------------------------
# Limpieza de tablas base
# -------------------------------------------------------------------

def prepare_categoria(categoria: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    - Elimina duplicados.
    - Elimina filas con id y categoria vacíos.
    - id -> int
    - categoria -> strip + upper
    - Crea o asegura 'OTHER' y devuelve su id.
    """
    categoria = categoria.copy()
    categoria["categoria"] = categoria["categoria"].astype(str).str.strip().str.upper()

    mask_empty = categoria["id"].isna() & (categoria["categoria"].replace("NAN", "").str.strip() == "")
    categoria = categoria[~mask_empty].drop_duplicates(subset=["id", "categoria"]).copy()

    # Asegurar fila OTHER
    if (categoria["categoria"] == "OTHER").any():
        other_cat_id = int(pd.to_numeric(categoria.loc[categoria["categoria"] == "OTHER", "id"]).iloc[0])
    else:
        max_id = pd.to_numeric(categoria["id"], errors="coerce").dropna().max()
        max_id = int(max_id) if pd.notna(max_id) else 0
        other_cat_id = max_id + 1
        categoria = pd.concat([
            categoria,
            pd.DataFrame([{"id": other_cat_id, "categoria": "OTHER"}])
        ], ignore_index=True)

    # id -> int robusto
    categoria["id"] = safe_int_cast(pd.to_numeric(categoria["id"], errors="coerce"), fill_value=other_cat_id)
    return categoria, other_cat_id


def prepare_marca(marca: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    - Elimina duplicados.
    - Elimina filas con id y marca vacíos.
    - id -> int
    - marca -> strip + upper
    - Crea o asegura 'OTHER' y devuelve su id.
    """
    marca = marca.copy()
    marca["marca"] = marca["marca"].astype(str).str.strip().str.upper()

    mask_empty = marca["id"].isna() & (marca["marca"].replace("NAN", "").str.strip() == "")
    marca = marca[~mask_empty].drop_duplicates(subset=["id", "marca"]).copy()

    # Asegurar fila OTHER
    if (marca["marca"] == "OTHER").any():
        other_brand_id = int(pd.to_numeric(marca.loc[marca["marca"] == "OTHER", "id"]).iloc[0])
    else:
        max_id = pd.to_numeric(marca["id"], errors="coerce").dropna().max()
        max_id = int(max_id) if pd.notna(max_id) else 0
        other_brand_id = max_id + 1
        marca = pd.concat([
            marca,
            pd.DataFrame([{"id": other_brand_id, "marca": "OTHER"}])
        ], ignore_index=True)

    marca["id"] = safe_int_cast(pd.to_numeric(marca["id"], errors="coerce"), fill_value=other_brand_id)
    return marca, other_brand_id


def prepare_producto(producto: pd.DataFrame, other_brand_id: int, other_cat_id: int) -> pd.DataFrame:
    """
    Reglas:
      - Elimina duplicados.
      - Elimina filas con categoria_id nulo y volumen == 0.
      - precio: imputar por mediana (categoria_id, volumen) -> luego por volumen -> global. Reemplaza 0 por NaN.
      - Si marca_id y categoria_id son nulos, asigna OTHER a ambos.
      - categoria_id nulo restante -> moda por (marca_id, volumen); si sigue nulo -> OTHER.
      - id, categoria_id, marca_id -> int
      - nombre -> strip + upper
    """
    producto = producto.copy().drop_duplicates()

    # Tipos
    for col in ["categoria_id", "marca_id", "volumen", "precio"]:
        if col in producto.columns:
            producto[col] = pd.to_numeric(producto[col], errors="coerce")

    # Eliminar (categoria_id NaN) & (volumen == 0)
    mask_drop = producto["categoria_id"].isna() & (producto["volumen"].fillna(0) == 0)
    producto = producto[~mask_drop].copy()

    # Imputación de precio
    grp_median = producto.groupby(["categoria_id", "volumen"])["precio"].transform("median")
    producto["precio"] = producto["precio"].fillna(grp_median)
    producto["precio"] = producto["precio"].replace(0, np.nan)
    producto["precio"] = producto["precio"].fillna(producto.groupby("volumen")["precio"].transform("median"))
    producto["precio"] = producto["precio"].fillna(producto["precio"].median())

    # BOTH NaN -> OTHER
    mask_both_na = producto["marca_id"].isna() & producto["categoria_id"].isna()
    if mask_both_na.any():
        producto.loc[mask_both_na, ["marca_id", "categoria_id"]] = [other_brand_id, other_cat_id]

    # categoria_id por moda (marca_id, volumen) y luego OTHER
    producto["categoria_id"] = producto["categoria_id"].where(
        producto["categoria_id"].notna(),
        group_mode_transform(producto, ["marca_id", "volumen"], "categoria_id")
    )
    producto["categoria_id"] = producto["categoria_id"].fillna(other_cat_id)

    # Casts y normalizaciones finales
    producto["id"] = safe_int_cast(pd.to_numeric(producto["id"], errors="coerce"))
    producto["categoria_id"] = safe_int_cast(pd.to_numeric(producto["categoria_id"], errors="coerce"), fill_value=other_cat_id)
    producto["marca_id"] = safe_int_cast(pd.to_numeric(producto["marca_id"], errors="coerce"), fill_value=other_brand_id)
    producto["nombre"] = producto["nombre"].astype(str).str.strip().str.upper()

    return producto


def prepare_cliente(cliente: pd.DataFrame, parse_mdY_func) -> pd.DataFrame:
    """
    - Drop columnas: nit, puesto, correo, telefono
    - Elimina duplicados y filas totalmente vacías
    - nombre = nombre + ' ' + apellido; drop 'apellido'
    - id -> int
    - nacimiento -> parse_mdY
    - edad en años cumplidos
    - Mayúsculas en: nombre, genero(F/M -> FEMALE/MALE), empresa, idioma, ciudad
    """
    cliente = cliente.copy()

    # Drop columnas
    cols_drop = ["nit", "puesto", "correo", "telefono"]
    cliente = cliente.drop(columns=[c for c in cols_drop if c in cliente.columns], errors="ignore")

    # Dups & vacías
    cliente = cliente.drop_duplicates().dropna(how="all").copy()

    # Nombre completo
    cliente["nombre"] = (
        cliente["nombre"].fillna("").astype(str).str.strip() + " " +
        cliente["apellido"].fillna("").astype(str).str.strip()
    ).str.strip()
    cliente = cliente.drop(columns=["apellido"], errors="ignore")

    # id
    cliente["id"] = safe_int_cast(pd.to_numeric(cliente["id"], errors="coerce"), fill_value=0)

    # nacimiento y edad
    cliente["nacimiento"] = cliente["nacimiento"].map(parse_mdY_func)
    today = pd.Timestamp.today().normalize()
    cliente["edad"] = np.floor((today - cliente["nacimiento"]).dt.days / 365.25).astype("Int64")

    # Transformaciones
    cliente["nombre"] = cliente["nombre"].astype(str).str.upper()
    if "genero" in cliente.columns:
        cliente["genero"] = cliente["genero"].astype(str).str.upper()
        cliente["genero"] = cliente["genero"].replace({"F": "FEMALE", "M": "MALE"})
    if "empresa" in cliente.columns:
        cliente["empresa"] = cliente["empresa"].astype(str).str.upper()
    if "idioma" in cliente.columns:
        cliente["idioma"] = cliente["idioma"].astype(str).str.upper()
    if "ciudad" in cliente.columns:
        cliente["ciudad"] = cliente["ciudad"].astype(str).str.upper()

    return cliente


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """
    - fecha = to_datetime(timestamp, unit='ms')
    - drop 'timestamp', duplicados y 'transactionid'
    - event -> upper
    """
    events = events.copy()
    events["fecha"] = pd.to_datetime(events["timestamp"], unit="ms", errors="coerce")
    events = events.drop(columns=["timestamp"], errors="ignore")
    events = events.drop_duplicates().copy()
    events = events.drop(columns=["transactionid"], errors="ignore")
    events["event"] = events["event"].astype(str).str.upper()
    return events

# -------------------------------------------------------------------
# Unificación (MERGEs) exactamente como indicaste
# -------------------------------------------------------------------

def enrich_producto_with_marca_categoria(producto: pd.DataFrame, marca: pd.DataFrame, categoria: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las uniones y drops solicitados para producto con marca y categoría.
    """
    # producto + marca
    tmp = pd.merge(producto, marca, left_on="marca_id", right_on="id", how="left", indicator=True)
    producto_m = tmp.drop(columns=["_merge", "id_y", "marca_id"]).copy()

    # producto + categoria
    tmp = pd.merge(producto_m, categoria, left_on="categoria_id", right_on="id", how="left", indicator=True)
    producto_mc = tmp.drop(columns=["_merge", "id", "categoria_id"]).copy()
    producto_mc = producto_mc.rename(columns={"id_x": "id"})

    return producto_mc


def merge_events_with_entities(events: pd.DataFrame, producto_enriq: pd.DataFrame, cliente: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera:
      - events1: events + producto (indicator renombrado a 'event+producto')
      - events2: events + cliente  (indicator renombrado a 'event+cliente')
      - eventos3: outer merge de ambos manteniendo ['visitorid','itemid']
    """
    events1 = pd.merge(events, producto_enriq, left_on="itemid", right_on="id", how="left", indicator=True)
    events1 = events1.drop(columns=["id"]).rename(columns={"_merge": "event+producto"})

    events2 = pd.merge(events, cliente, left_on="visitorid", right_on="id", how="left", indicator=True)
    events2 = events2.drop(columns=["id"]).rename(columns={"_merge": "event+cliente"})

    cols_to_use = [c for c in events2.columns if c not in events1.columns or c in ["visitorid", "itemid"]]
    eventos3 = pd.merge(events1, events2[cols_to_use], on=["visitorid", "itemid"], how="outer")

    return events1, events2, eventos3

# -------------------------------------------------------------------
# Orquestador del pipeline
# -------------------------------------------------------------------

def run_pipeline(files: dict,
                 parse_mdY_func,
                 save_dir: str = "processed_data") -> dict:
    """
    Carga, limpia, enriquece y exporta todo el pipeline.
    Requiere:
      - files: dict con rutas (categoria, cliente, events, marca, producto)
      - parse_mdY_func: referencia a tu función parse_mdY
      - save_dir: carpeta de salida CSVs
    Devuelve dict con DataFrames y metadatos útiles.
    """
    # 1) Cargar
    categoria = load_csv_robusto(files["categoria"])
    cliente   = load_csv_robusto(files["cliente"])
    events    = load_csv_robusto(files["events"])
    marca     = load_csv_robusto(files["marca"])
    producto  = load_csv_robusto(files["producto"])

    # 2) Limpiar dimensión
    categoria, other_cat_id   = prepare_categoria(categoria)
    marca, other_brand_id     = prepare_marca(marca)
    producto                  = prepare_producto(producto, other_brand_id, other_cat_id)
    cliente                   = prepare_cliente(cliente, parse_mdY_func)
    events                    = prepare_events(events)

    # 3) Enriquecer producto con descripciones
    producto_enriq = enrich_producto_with_marca_categoria(producto, marca, categoria)

    # 4) Unir eventos
    events1, events2, eventos3 = merge_events_with_entities(events, producto_enriq, cliente)

    # 5) Exportar
    os.makedirs(save_dir, exist_ok=True)
    cliente.to_csv(os.path.join(save_dir, "cliente.csv"), index=False)
    categoria.to_csv(os.path.join(save_dir, "categoria.csv"), index=False)
    marca.to_csv(os.path.join(save_dir, "marca.csv"), index=False)
    producto_enriq.to_csv(os.path.join(save_dir, "producto.csv"), index=False)
    events.to_csv(os.path.join(save_dir, "evento.csv"), index=False)

    events1.to_csv(os.path.join(save_dir, "eventos_productos.csv"), index=False)
    events2.to_csv(os.path.join(save_dir, "eventos_clientes.csv"), index=False)
    eventos3.to_csv(os.path.join(save_dir, "events_productos_clientes.csv"), index=False)

    return {
        "categoria": categoria,
        "marca": marca,
        "producto": producto_enriq,
        "cliente": cliente,
        "events": events,
        "events1": events1,
        "events2": events2,
        "eventos3": eventos3,
        "OTHER_IDS": {"categoria": other_cat_id, "marca": other_brand_id},
        "save_dir": save_dir
    }
    
files = {
    "categoria": "data/categoria.csv",
    "cliente": "data/cliente.csv",
    "events": "data/events.csv",
    "marca": "data/marca.csv",
    "producto": "data/producto.csv",
}

result = run_pipeline(files, parse_mdY_func=parse_mdY, save_dir="prueba_dir")
print(result["OTHER_IDS"])