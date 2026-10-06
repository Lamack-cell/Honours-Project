import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 4))
ax.axis('off')

# Define box properties
box_style = "round,pad=0.6"
props_blue = dict(boxstyle=box_style, facecolor='#e6f2ff', edgecolor='#1f77b4', linewidth=2)
props_orange = dict(boxstyle=box_style, facecolor='#fff0e6', edgecolor='#ff7f0e', linewidth=2)
props_green = dict(boxstyle=box_style, facecolor='#e6ffe6', edgecolor='#2ca02c', linewidth=2)

y_center = 0.5
x_pos = [0.125, 0.375, 0.625, 0.875]

texts = [
    "1. Broad Market Panel\nRaw OHLCV Data for\n62 JSE-Listed Equities\n(153,016 obs)",
    "2. Liquidity Filtration\n63-Day Rolling Average\nof Daily Value Traded\n(Close x Volume)",
    "3. Quarterly Rebalancing\nRank assets & select\nthe Top 40 most liquid\nper quarter",
    "4. Active Universe ($A_t$)\nSurvivorship-Bias-Free\nRolling Panel\n(105,283 obs)"
]

colors = [props_blue, props_orange, props_orange, props_green]

for i in range(4):
    ax.text(x_pos[i], y_center, texts[i], ha="center", va="center", size=11,
            bbox=colors[i], fontweight='bold')

# Draw arrows
arrow_props = dict(facecolor='black', edgecolor='black', width=2, headwidth=8, shrink=0.1)

for i in range(3):
    ax.annotate('', xy=(x_pos[i+1] - 0.12, y_center), xytext=(x_pos[i] + 0.12, y_center),
                arrowprops=arrow_props)

plt.title("Figure 4.1: Active Universe Filtration Pipeline", fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()

# Save the plot
plt.savefig("Figure_4_1_Pipeline_Horizontal.png", dpi=300, bbox_inches='tight')
print("Success: Horizontal flowchart saved as 'Figure_4_1_Pipeline_Horizontal.png'")