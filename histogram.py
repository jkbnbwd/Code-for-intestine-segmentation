import matplotlib.pyplot as plt

# Dice scores and SD values
dice_scores = [79.14, 78.33, 78.11, 79.42]
sd_values = [1.79, 1.99, 1.30, 1.10]

plt.rcParams.update({'font.size': 13})


# X-axis values
x_values = list(range(1, len(dice_scores) + 1))

# Create the histogram
plt.figure(figsize=(8, 6))
plt.bar(x_values, dice_scores, tick_label=x_values, align='center', alpha=0.7, label='Dice Score')
plt.xlabel('Experiment')
plt.ylabel('Dice Score')
plt.title('Dice Score Histogram with Standard Deviation')
plt.xticks(x_values)

# Add vertical lines for SD values
for i in range(len(dice_scores)):
    plt.axvline(x=x_values[i], color='red', linestyle='dashed', linewidth=1)
    plt.text(x_values[i] + 0.05, dice_scores[i] + 1, f'SD {sd_values[i]:.2f}', color='red')

plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('dice_score_histogram.png')
plt.show()