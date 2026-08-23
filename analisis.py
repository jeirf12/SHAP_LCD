import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine
from matplotlib.ticker import FuncFormatter


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

OUTPUT_DIR = "graficas_exploracion"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# ARCHIVOS CON LOS NOMBRES DE LAS CLASES
# ============================================================

IRIS_CLASS_FILE = "CLASS_NAMES_IRIS.json"
VEHICLE_CLASS_FILE = "CLASS_NAMES_VEHICLE.json"


# ============================================================
# FUNCIÓN PARA CARGAR NOMBRES DE CLASES
# ============================================================

def cargar_nombres_clases(ruta):

    with open(ruta, "r", encoding="utf-8") as archivo:
        nombres = json.load(archivo)

    # Convertir las claves del JSON de string a entero
    # Ejemplo:
    # {"0": "setosa", "1": "versicolor"}
    #
    # se convierte en:
    # {0: "setosa", 1: "versicolor"}

    nombres = {
        int(clave): valor
        for clave, valor in nombres.items()
    }

    return nombres


# ============================================================
# CARGAR LOS NOMBRES DE LAS CLASES
# ============================================================

iris_class_names = cargar_nombres_clases(
    IRIS_CLASS_FILE
)

vehicle_class_names = cargar_nombres_clases(
    VEHICLE_CLASS_FILE
)


print("\nNombres de clases Iris:")
print(iris_class_names)

print("\nNombres de clases Vehicle:")
print(vehicle_class_names)


# ============================================================
# FUNCIÓN PARA GUARDAR GRÁFICAS
# ============================================================

def guardar_grafica(nombre):

    ruta = os.path.join(
        OUTPUT_DIR,
        nombre
    )

    plt.tight_layout()

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"Gráfica guardada: {ruta}"
    )


# ============================================================
# FUNCIÓN PARA ANALIZAR DATOS FALTANTES
# ============================================================

def analizar_faltantes(
    df,
    nombre_dataset
):

    print("\n" + "=" * 70)
    print(
        f"ANÁLISIS DE DATOS FALTANTES - "
        f"{nombre_dataset.upper()}"
    )
    print("=" * 70)

    nulos = df.isnull().sum()

    total_nulos = nulos.sum()

    print(
        f"Total de valores NaN: {total_nulos}"
    )

    columnas_con_nulos = (
        nulos[nulos > 0]
        .sort_values(
            ascending=False
        )
    )

    if len(columnas_con_nulos) == 0:

        print(
            "No se encontraron valores nulos."
        )

        return

    print(
        "\nColumnas con valores nulos:"
    )

    print(
        columnas_con_nulos
    )

    plt.figure(
        figsize=(12, 6)
    )

    columnas = (
        columnas_con_nulos.index
    )

    valores = (
        columnas_con_nulos.values
    )

    bars = plt.barh(
        columnas,
        valores
    )

    plt.title(
        f"Valores faltantes - "
        f"{nombre_dataset}"
    )

    plt.xlabel(
        "Cantidad de valores faltantes"
    )

    plt.ylabel(
        "Variable"
    )

    for bar, value in zip(
        bars,
        valores
    ):

        plt.text(
            bar.get_width(),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:,}".replace(
                ",", "."
            ),
            va="center"
        )

    guardar_grafica(
        f"{nombre_dataset.lower().replace(' ', '_')}"
        "_datos_faltantes.png"
    )


# ============================================================
# INFORMACIÓN GENERAL
# ============================================================

def informacion_general(
    df,
    nombre_dataset,
    target=None,
    class_names=None
):

    print("\n" + "=" * 70)

    print(
        f"INFORMACIÓN GENERAL - "
        f"{nombre_dataset.upper()}"
    )

    print("=" * 70)

    print(
        f"Registros: {df.shape[0]}"
    )

    print(
        f"Atributos totales: "
        f"{df.shape[1]}"
    )

    if target is not None:

        print(
            f"Variable objetivo: "
            f"{target}"
        )

        print(
            f"Atributos predictivos: "
            f"{df.shape[1] - 1}"
        )

        cantidad_clases = (
            df[target].nunique()
        )

        print(
            f"Cantidad de clases: "
            f"{cantidad_clases}"
        )

        print(
            "\nDistribución de clases:"
        )

        conteo = (
            df[target]
            .value_counts()
            .sort_index()
        )

        for clase, cantidad in (
            conteo.items()
        ):

            nombre = (
                class_names.get(
                    int(clase),
                    str(clase)
                )
                if class_names
                else str(clase)
            )

            print(
                f"  {clase} - {nombre}: "
                f"{cantidad:,}".replace(
                    ",", "."
                )
            )


# ============================================================
# GRÁFICA DE CLASES
# ============================================================

def grafica_clases(
    df,
    target,
    nombre_dataset,
    class_names=None
):

    if target not in df.columns:

        print(
            f"No se encontró la variable "
            f"'{target}' en "
            f"{nombre_dataset}."
        )

        return

    conteo = (
        df[target]
        .value_counts()
        .sort_index()
    )

    # Convertir códigos a nombres
    etiquetas = []

    for clase in conteo.index:

        try:

            clase_num = int(clase)

            nombre = (
                class_names.get(
                    clase_num,
                    str(clase)
                )
                if class_names
                else str(clase)
            )

        except:

            nombre = str(clase)

        etiquetas.append(
            nombre
        )

    plt.figure(
        figsize=(9, 6)
    )

    bars = sns.barplot(
        x=etiquetas,
        y=conteo.values
    )

    plt.title(
        f"Distribución de clases - "
        f"{nombre_dataset}"
    )

    plt.xlabel(
        "Clase"
    )

    plt.ylabel(
        "Cantidad de registros"
    )

    for bar, valor in zip(
        bars.patches,
        conteo.values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{valor:,}".replace(
                ",", "."
            ),
            ha="center",
            va="bottom"
        )

    guardar_grafica(
        f"{nombre_dataset.lower().replace(' ', '_')}"
        "_distribucion_clases.png"
    )


# ============================================================
# DISTRIBUCIÓN DE ATRIBUTOS
# ============================================================

def grafica_atributos_numericos(
    df,
    nombre_dataset,
    columnas=None,
    cantidad=4
):

    columnas_numericas = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    if columnas is None:

        columnas = (
            columnas_numericas[:cantidad]
        )

    else:

        columnas = [
            col
            for col in columnas
            if col in df.columns
        ]

    if len(columnas) == 0:

        print(
            f"No se encontraron "
            f"atributos numéricos "
            f"para {nombre_dataset}."
        )

        return

    print(
        f"\nAtributos seleccionados "
        f"para {nombre_dataset}:"
    )

    for col in columnas:

        print(
            f" - {col}"
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9)
    )

    axes = axes.flatten()

    for i, columna in enumerate(
        columnas
    ):

        sns.histplot(
            df[columna].dropna(),
            kde=True,
            ax=axes[i]
        )

        axes[i].set_title(
            f"Distribución de {columna}"
        )

        axes[i].set_xlabel(
            columna
        )

        axes[i].set_ylabel(
            "Frecuencia"
        )

    for i in range(
        len(columnas),
        len(axes)
    ):

        axes[i].axis("off")

    plt.suptitle(
        f"Distribución de atributos - "
        f"{nombre_dataset}",
        fontsize=15
    )

    guardar_grafica(
        f"{nombre_dataset.lower().replace(' ', '_')}"
        "_distribucion_atributos.png"
    )


# ============================================================
# BOXPLOTS
# ============================================================

def grafica_boxplots(
    df,
    nombre_dataset,
    columnas=None,
    cantidad=4
):

    columnas_numericas = (
        df.select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    if columnas is None:

        columnas = (
            columnas_numericas[:cantidad]
        )

    else:

        columnas = [
            col
            for col in columnas
            if col in df.columns
        ]

    if len(columnas) == 0:

        return

    fig, axes = plt.subplots(
        1,
        len(columnas),
        figsize=(16, 6)
    )

    if len(columnas) == 1:

        axes = [axes]

    for ax, columna in zip(
        axes,
        columnas
    ):

        sns.boxplot(
            y=df[columna],
            ax=ax
        )

        ax.set_title(
            columna
        )

        ax.set_ylabel(
            "Valor"
        )

    plt.suptitle(
        f"Distribución y posibles "
        f"valores atípicos - "
        f"{nombre_dataset}",
        fontsize=15
    )

    guardar_grafica(
        f"{nombre_dataset.lower().replace(' ', '_')}"
        "_boxplots.png"
    )


# ============================================================
# 1. IRIS
# ============================================================

print("\n\n")
print("#" * 80)
print("# DATASET IRIS")
print("#" * 80)

iris = load_iris(
    as_frame=True
)

df_iris = iris.frame.copy()

target_iris = "target"

informacion_general(
    df_iris,
    "Iris",
    target_iris,
    iris_class_names
)

analizar_faltantes(
    df_iris,
    "Iris"
)

grafica_clases(
    df_iris,
    target_iris,
    "Iris",
    iris_class_names
)

grafica_atributos_numericos(
    df_iris,
    "Iris",
    columnas=iris.feature_names
)

grafica_boxplots(
    df_iris,
    "Iris",
    columnas=iris.feature_names
)


# ============================================================
# 2. WINE
# ============================================================

print("\n\n")
print("#" * 80)
print("# DATASET WINE")
print("#" * 80)

wine = load_wine(
    as_frame=True
)

df_wine = wine.frame.copy()

target_wine = "target"

# Scikit-learn proporciona directamente
# los nombres de las clases de Wine.

wine_class_names = {
    i: nombre
    for i, nombre in enumerate(
        wine.target_names
    )
}

print(
    "\nNombres de clases Wine:"
)

print(
    wine_class_names
)

informacion_general(
    df_wine,
    "Wine",
    target_wine,
    wine_class_names
)

analizar_faltantes(
    df_wine,
    "Wine"
)

grafica_clases(
    df_wine,
    target_wine,
    "Wine",
    wine_class_names
)

columnas_wine = [
    "alcohol",
    "malic_acid",
    "color_intensity",
    "proline"
]

grafica_atributos_numericos(
    df_wine,
    "Wine",
    columnas=columnas_wine
)

grafica_boxplots(
    df_wine,
    "Wine",
    columnas=columnas_wine
)


# ============================================================
# 3. VEHICLE
# ============================================================

print("\n\n")
print("#" * 80)
print("# DATASET VEHICLE")
print("#" * 80)

vehicle_path = "vehicle.csv"

df_vehicle = pd.read_csv(
    vehicle_path,
    sep=None,
    engine="python"
)

print(
    "\nPrimeras filas:"
)

print(
    df_vehicle.head()
)

print(
    "\nColumnas:"
)

print(
    df_vehicle.columns.tolist()
)


# ------------------------------------------------------------
# Identificar variable objetivo
# ------------------------------------------------------------

posibles_targets = [
    "class",
    "Class",
    "target",
    "Target",
    "CLASS"
]

target_vehicle = None

for columna in posibles_targets:

    if columna in df_vehicle.columns:

        target_vehicle = columna

        break


if target_vehicle is None:

    target_vehicle = (
        df_vehicle.columns[-1]
    )

    print(
        "\nNo se identificó "
        "automáticamente la clase."
    )

    print(
        f"Se utilizará la última columna: "
        f"{target_vehicle}"
    )


informacion_general(
    df_vehicle,
    "Vehicle",
    target_vehicle,
    vehicle_class_names
)

analizar_faltantes(
    df_vehicle,
    "Vehicle"
)

grafica_clases(
    df_vehicle,
    target_vehicle,
    "Vehicle",
    vehicle_class_names
)


columnas_vehicle = (
    df_vehicle
    .drop(
        columns=[
            target_vehicle
        ]
    )
    .select_dtypes(
        include="number"
    )
    .columns
    .tolist()
)

columnas_vehicle = (
    columnas_vehicle[:4]
)

grafica_atributos_numericos(
    df_vehicle,
    "Vehicle",
    columnas=columnas_vehicle
)

grafica_boxplots(
    df_vehicle,
    "Vehicle",
    columnas=columnas_vehicle
)


# ============================================================
# 4. TRABAJO INFANTIL
# ============================================================

print("\n\n")
print("#" * 80)
print("# DATASET TRABAJO INFANTIL")
print("#" * 80)

trabajo_path = (
    "df_Menores_dumy.csv"
)

df_trabajo = pd.read_csv(
    trabajo_path,
    sep=";"
)

print(
    "\nPrimeras filas:"
)

print(
    df_trabajo.head()
)

print(
    f"\nRegistros: "
    f"{df_trabajo.shape[0]}"
)

print(
    f"Atributos: "
    f"{df_trabajo.shape[1]}"
)


# ------------------------------------------------------------
# Datos faltantes
# ------------------------------------------------------------

analizar_faltantes(
    df_trabajo,
    "Trabajo Infantil"
)


# ------------------------------------------------------------
# Valores cero
# ------------------------------------------------------------

print(
    "\n\nAnálisis de valores cero:"
)

cols_dummy = [
    c
    for c in df_trabajo.columns
    if " = " in c or "=" in c
]

cols_numericas = [
    c
    for c in df_trabajo
    .select_dtypes(
        include="number"
    )
    .columns
    if c not in cols_dummy
]

ceros = {
    col:
    (
        df_trabajo[col] == 0
    ).sum()

    for col in cols_numericas

    if (
        df_trabajo[col] == 0
    ).sum() > 0
}

for col, cantidad in (
    ceros.items()
):

    porcentaje = (
        cantidad /
        len(df_trabajo) *
        100
    )

    print(
        f"{col}: "
        f"{cantidad:,} "
        f"({porcentaje:.2f}%)"
        .replace(",", ".")
    )


# ------------------------------------------------------------
# Edad
# ------------------------------------------------------------

print(
    "\n\nDistribución de edades:"
)

print(
    df_trabajo["edad"].describe()
)

plt.figure(
    figsize=(10, 6)
)

sns.histplot(
    df_trabajo["edad"],
    kde=True
)

plt.title(
    "Distribución de edades - "
    "Trabajo Infantil"
)

plt.xlabel(
    "Edad"
)

plt.ylabel(
    "Frecuencia"
)

guardar_grafica(
    "trabajo_infantil_distribucion_edades.png"
)


# ------------------------------------------------------------
# Género
# ------------------------------------------------------------

print(
    "\n\nDistribución de género:"
)

genero_counts = pd.Series({

    "Hombre":
    df_trabajo[
        "sexo = hombre"
    ].sum(),

    "Mujer":
    df_trabajo[
        "sexo = mujer"
    ].sum()
})

print(
    genero_counts
)

plt.figure(
    figsize=(8, 6)
)

bars = sns.barplot(
    x=genero_counts.index,
    y=genero_counts.values
)

plt.title(
    "Distribución de género - "
    "Trabajo Infantil"
)

plt.xlabel(
    "Género"
)

plt.ylabel(
    "Cantidad"
)

for bar, valor in zip(
    bars.patches,
    genero_counts.values
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height(),
        f"{valor:,}".replace(
            ",", "."
        ),
        ha="center",
        va="bottom"
    )

guardar_grafica(
    "trabajo_infantil_distribucion_genero.png"
)


# ------------------------------------------------------------
# Formato COP
# ------------------------------------------------------------

def formato_cop(
    x,
    pos
):

    return (
        f"${x:,.0f}"
        .replace(",", ".")
    )


fmt_cop = FuncFormatter(
    formato_cop
)


# ------------------------------------------------------------
# Ingresos
# ------------------------------------------------------------

print(
    "\n\nDistribución de ingresos:"
)

print(
    df_trabajo[
        "i_hogar"
    ].describe()
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

sns.histplot(
    df_trabajo[
        "i_hogar"
    ],
    kde=True,
    ax=axes[0]
)

axes[0].set_title(
    "Distribución de ingresos familiares"
)

axes[0].set_xlabel(
    "Ingreso mensual (COP)"
)

axes[0].set_ylabel(
    "Frecuencia"
)

axes[0].xaxis.set_major_formatter(
    fmt_cop
)

axes[0].tick_params(
    axis="x",
    rotation=45
)

sns.boxplot(
    y=df_trabajo[
        "i_hogar"
    ],
    ax=axes[1]
)

axes[1].set_title(
    "Boxplot de ingresos familiares"
)

axes[1].set_ylabel(
    "Ingreso mensual (COP)"
)

axes[1].yaxis.set_major_formatter(
    fmt_cop
)

guardar_grafica(
    "trabajo_infantil_distribucion_ingresos.png"
)


# ------------------------------------------------------------
# Gastos
# ------------------------------------------------------------

print(
    "\n\nDistribución de gastos:"
)

print(
    df_trabajo[
        "i_ugasto"
    ].describe()
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

sns.histplot(
    df_trabajo[
        "i_ugasto"
    ],
    kde=True,
    ax=axes[0]
)

axes[0].set_title(
    "Distribución de gastos familiares"
)

axes[0].set_xlabel(
    "Gasto mensual (COP)"
)

axes[0].set_ylabel(
    "Frecuencia"
)

axes[0].xaxis.set_major_formatter(
    fmt_cop
)

axes[0].tick_params(
    axis="x",
    rotation=45
)

sns.boxplot(
    y=df_trabajo[
        "i_ugasto"
    ],
    ax=axes[1]
)

axes[1].set_title(
    "Boxplot de gastos familiares"
)

axes[1].set_ylabel(
    "Gasto mensual (COP)"
)

axes[1].yaxis.set_major_formatter(
    fmt_cop
)

guardar_grafica(
    "trabajo_infantil_distribucion_gastos.png"
)


# ============================================================
# RANGOS DE INGRESOS Y GASTOS
# ============================================================

rangos = [
    0,
    500_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    50_000_000,
    float("inf")
]

etiquetas = [
    "$0-500K",
    "$500K-1M",
    "$1M-2M",
    "$2M-5M",
    "$5M-10M",
    "$10M-50M",
    "$50M+"
]

df_trabajo[
    "rango_ingresos"
] = pd.cut(
    df_trabajo[
        "i_hogar"
    ],
    bins=rangos,
    labels=etiquetas,
    right=False
)

df_trabajo[
    "rango_gastos"
] = pd.cut(
    df_trabajo[
        "i_ugasto"
    ],
    bins=rangos,
    labels=etiquetas,
    right=False
)

conteo_ing = (
    df_trabajo[
        "rango_ingresos"
    ]
    .value_counts()
    .sort_index()
)

conteo_gas = (
    df_trabajo[
        "rango_gastos"
    ]
    .value_counts()
    .sort_index()
)

print(
    "\nDistribución por rangos de ingresos:"
)

print(
    conteo_ing
)

print(
    "\nDistribución por rangos de gastos:"
)

print(
    conteo_gas
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7)
)

sns.barplot(
    x=conteo_ing.index,
    y=conteo_ing.values,
    ax=axes[0]
)

axes[0].set_title(
    "Distribución de ingresos por rangos"
)

axes[0].set_xlabel(
    "Rango de ingresos (COP)"
)

axes[0].set_ylabel(
    "Cantidad de familias"
)

axes[0].tick_params(
    axis="x",
    rotation=45
)

for i, valor in enumerate(
    conteo_ing.values
):

    axes[0].text(
        i,
        valor,
        f"{valor:,}".replace(
            ",", "."
        ),
        ha="center",
        va="bottom"
    )

sns.barplot(
    x=conteo_gas.index,
    y=conteo_gas.values,
    ax=axes[1]
)

axes[1].set_title(
    "Distribución de gastos por rangos"
)

axes[1].set_xlabel(
    "Rango de gastos (COP)"
)

axes[1].set_ylabel(
    "Cantidad de familias"
)

axes[1].tick_params(
    axis="x",
    rotation=45
)

for i, valor in enumerate(
    conteo_gas.values
):

    axes[1].text(
        i,
        valor,
        f"{valor:,}".replace(
            ",", "."
        ),
        ha="center",
        va="bottom"
    )

guardar_grafica(
    "trabajo_infantil_distribucion_por_rangos.png"
)


# ============================================================
# CONSOLIDADO
# ============================================================

print("\n\n")
print("=" * 80)
print("CONSOLIDADO DE DATASETS")
print("=" * 80)

resumen = pd.DataFrame({

    "Dataset": [
        "Iris",
        "Wine",
        "Vehicle",
        "Trabajo Infantil"
    ],

    "Registros": [
        df_iris.shape[0],
        df_wine.shape[0],
        df_vehicle.shape[0],
        df_trabajo.shape[0]
    ],

    "Atributos": [
        df_iris.shape[1] - 1,
        df_wine.shape[1] - 1,
        df_vehicle.shape[1] - 1,
        df_trabajo.shape[1]
    ],

    "Clases": [
        df_iris[
            target_iris
        ].nunique(),

        df_wine[
            target_wine
        ].nunique(),

        df_vehicle[
            target_vehicle
        ].nunique(),

        "N/A"
    ]
})

print(
    resumen.to_string(
        index=False
    )
)

resumen.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "consolidado_datasets.csv"
    ),
    index=False
)

print(
    "\nAnálisis finalizado."
)

print(
    f"Las gráficas se encuentran "
    f"en: {OUTPUT_DIR}"
)
