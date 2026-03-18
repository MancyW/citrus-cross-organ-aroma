print("\n>>> [Step5] 开始绘制 A2 图：PNG/TIF + SVG/PDF（矢量）")

if len(final_vocs) == 0:

    print("!!! 无关键 VOC，无法绘图")

else:

    stage_colors = {

        "S1": "#DCD7EB",

        "S2": "#FCE6CF",

        "S3": "#D5EAD9",

        "S4": "#7DC69B"

    }

    df_z = df.copy()

    df_z[final_vocs] = (df[final_vocs] - df[final_vocs].mean()) / df[final_vocs].std()

    plot_df = df_z[["Stage"] + final_vocs].melt(

        id_vars="Stage", var_name="VOC", value_name="Z"

    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes = axes.flatten()

    target_axes = [0, 2]

    for ax in [axes[1], axes[3]]:

        ax.axis("off")

    for idx, voc in enumerate(final_vocs):

        if idx >= 2:

            break

        ax = axes[target_axes[idx]]

        sub = plot_df[plot_df["VOC"] == voc]

        sns.swarmplot(

            data=sub,

            x="Stage",

            y="Z",

            order=stage_order,

            palette=stage_colors,

            size=6,

            alpha=1.0,

            linewidth=0,

            ax=ax

        )

        means = sub.groupby("Stage")["Z"].mean().reindex(stage_order)

        sds   = sub.groupby("Stage")["Z"].std().reindex(stage_order)

        x = np.arange(len(stage_order))

        ax.fill_between(

            x, means - sds, means + sds,

            color="#CCCCCC", alpha=0.20

        )

        ax.plot(

            x, means.values,

            color="black", linewidth=2.2,

            marker="o", markersize=8,

            markerfacecolor="white",

            markeredgecolor="black"

        )

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(1.2)

        ax.spines["bottom"].set_linewidth(1.2)

        ax.set_xticks(x)

        ax.set_xticklabels(stage_order, fontsize=12)

        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        ax.tick_params(axis="y", labelsize=12)

        ax.set_ylim(-2, 4)

        ax.set_yticks(np.linspace(-1.5, 3.5, 5))

        ax.set_title(voc, fontsize=16, pad=10)

        ax.set_xlabel("Stage", fontsize=14)

        ax.set_ylabel("Z-score", fontsize=14)

    plt.tight_layout()

    for ext in ["png", "tif", "svg", "pdf"]:

        out_path = OUT_DIR / f"Fig1E_keyVOC_trends_swarm_2*1.{ext}"

        plt.savefig(out_path, dpi=600)

        print(">>> 输出文件:", out_path)

print("\n>>> [Done] Fig1E（Leaf）v5 — 左侧两图版完成！")
