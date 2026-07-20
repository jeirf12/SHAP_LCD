import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('df_Menores_dumy.csv', sep=';')
print(df.head())

# Análisis de datos faltantes: nulos, vacíos y ceros sospechosos
print("\n\nAnálisis de datos faltantes:")
nulos = df.isnull().sum()
print(f"Total valores NaN: {nulos.sum()}")

# Columnas numéricas NO dummy (las dummy son 0/1 y los ceros son válidos)
cols_dummy = [c for c in df.columns if ' = ' in c or '=' in c]
cols_numericas = [c for c in df.select_dtypes(include='number').columns if c not in cols_dummy]
print(f"\nColumnas numéricas no-dummy: {cols_numericas}")

# Contar ceros en columnas numéricas relevantes (posibles datos faltantes)
ceros = {col: (df[col] == 0).sum() for col in cols_numericas if (df[col] == 0).sum() > 0}
print("\nCeros por columna numérica (posibles datos faltantes):")
for col, count in ceros.items():
    pct = count / len(df) * 100
    print(f"  {col}: {count} ({pct:.1f}%)")

# Gráfica: ceros sospechosos en columnas numéricas clave
if ceros:
    ceros_series = pd.Series(ceros).sort_values(ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Barras horizontales con cantidad de ceros
    bars = axes[0].barh(ceros_series.index, ceros_series.values, color='tomato')
    axes[0].set_title('Valores en cero por columna numérica\n(posibles datos faltantes)')
    axes[0].set_xlabel('Cantidad de registros con valor 0')
    for bar, v in zip(bars, ceros_series.values):
        pct = v / len(df) * 100
        axes[0].text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                     f'{v:,} ({pct:.1f}%)'.replace(',', '.'), va='center', fontsize=9)

    # Heatmap de completitud (% de datos válidos != 0 y != NaN)
    completitud = pd.DataFrame({
        'Columna': cols_numericas,
        '% Válidos': [((df[c] != 0) & df[c].notna()).sum() / len(df) * 100 for c in cols_numericas]
    }).sort_values('% Válidos')
    sns.heatmap(
        completitud.set_index('Columna')[['% Válidos']],
        annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
        cbar_kws={'label': '% datos válidos (no nulos, no cero)'},
        ax=axes[1]
    )
    axes[1].set_title('Completitud de columnas numéricas\n(% valores != 0 y != NaN)')

    plt.tight_layout()
    plt.savefig('analisis_datos_faltantes.png')
    plt.show()
else:
    print("No se encontraron ceros sospechosos en columnas numéricas.")

# Analizar la distribución de edades
print("\n\nDistribución de edades:")
print(df['edad'].describe())

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['edad'], kde=True, ax=ax)
ax.set_title('Distribución de edades')
ax.set_xlabel('Edad')
ax.set_ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('distribucion_edades.png')
plt.show()

# Analizar la distribución de género
print("\n\nDistribución de género:")
print(df['sexo = hombre'].value_counts())
print(df['sexo = mujer'].value_counts())

genero_counts = pd.Series({
    'Hombre': df['sexo = hombre'].sum(),
    'Mujer': df['sexo = mujer'].sum()
})
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=genero_counts.index, y=genero_counts.values, ax=ax)
ax.set_title('Distribución de género')
ax.set_ylabel('Cantidad')
ax.set_xlabel('Género')
plt.tight_layout()
plt.savefig('distribucion_genero.png')
plt.show()

# Formateador para pesos colombianos
def formato_cop(x, pos):
    return f'${x:,.0f}'.replace(',', '.')

from matplotlib.ticker import FuncFormatter
fmt_cop = FuncFormatter(formato_cop)

# Analizar la distribución de ingresos totales familiares
print("\n\nDistribución de ingresos totales familiares:")
print(df['i_hogar'].head())
print(df['i_hogar'].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(df['i_hogar'], kde=True, ax=axes[0])
axes[0].set_title('Distribución de ingresos familiares')
axes[0].set_xlabel('Ingreso mensual (COP)')
axes[0].set_ylabel('Frecuencia')
axes[0].xaxis.set_major_formatter(fmt_cop)
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(y=df['i_hogar'], ax=axes[1])
axes[1].set_title('Boxplot de ingresos familiares')
axes[1].set_ylabel('Ingreso mensual (COP)')
axes[1].yaxis.set_major_formatter(fmt_cop)
plt.tight_layout()
plt.savefig('distribucion_ingresos.png')
plt.show()

# Analizar la distribución de gastos totales familiares
print("\n\nDistribución de gastos totales familiares:")
print(df['i_ugasto'].head())
print(df['i_ugasto'].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.histplot(df['i_ugasto'], kde=True, ax=axes[0])
axes[0].set_title('Distribución de gastos familiares')
axes[0].set_xlabel('Gasto mensual (COP)')
axes[0].set_ylabel('Frecuencia')
axes[0].xaxis.set_major_formatter(fmt_cop)
axes[0].tick_params(axis='x', rotation=45)

sns.boxplot(y=df['i_ugasto'], ax=axes[1])
axes[1].set_title('Boxplot de gastos familiares')
axes[1].set_ylabel('Gasto mensual (COP)')
axes[1].yaxis.set_major_formatter(fmt_cop)
plt.tight_layout()
plt.savefig('distribucion_gastos.png')
plt.show()

# Análisis de distribución por rangos de ingresos y gastos
rangos = [0, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 50_000_000, float('inf')]
etiquetas = ['$0-500K', '$500K-1M', '$1M-2M', '$2M-5M', '$5M-10M', '$10M-50M', '$50M+']

df['rango_ingresos'] = pd.cut(df['i_hogar'], bins=rangos, labels=etiquetas, right=False)
df['rango_gastos'] = pd.cut(df['i_ugasto'], bins=rangos, labels=etiquetas, right=False)

print("\n\nDistribución por rangos de ingresos:")
print(df['rango_ingresos'].value_counts().sort_index())
print("\nDistribución por rangos de gastos:")
print(df['rango_gastos'].value_counts().sort_index())

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Distribución por rangos - Ingresos
conteo_ing = df['rango_ingresos'].value_counts().sort_index()
sns.barplot(x=conteo_ing.index, y=conteo_ing.values, ax=axes[0], color='steelblue')
axes[0].set_title('Distribución de ingresos por rangos')
axes[0].set_xlabel('Rango de ingresos (COP)')
axes[0].set_ylabel('Cantidad de familias')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(conteo_ing.values):
    axes[0].text(i, v + 50, f'{v:,}'.replace(',', '.'), ha='center', fontsize=9)

# Distribución por rangos - Gastos
conteo_gas = df['rango_gastos'].value_counts().sort_index()
sns.barplot(x=conteo_gas.index, y=conteo_gas.values, ax=axes[1], color='coral')
axes[1].set_title('Distribución de gastos por rangos')
axes[1].set_xlabel('Rango de gastos (COP)')
axes[1].set_ylabel('Cantidad de familias')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(conteo_gas.values):
    axes[1].text(i, v + 50, f'{v:,}'.replace(',', '.'), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('distribucion_por_rangos.png')
plt.show()

# Limpiar columnas temporales
df.drop(columns=['rango_ingresos', 'rango_gastos'], inplace=True)