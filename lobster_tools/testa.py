from absl import app, flags, logging

# flags.DEFINE_multi_float(
#     "markouts",
#     [0.5, 1.0, 2.0, 4.0],
#     "Markouts as a function of the .",
# )

flags.DEFINE_list(
    "markouts",
    [0.5, 1.0, 2.0, 4.0],
    "Markouts as a function of the.",
)

FLAGS = flags.FLAGS

def main(_):
    markouts = FLAGS.markouts
    print(markouts)
    print(type(markouts))
    print(type(markouts[0]))

if __name__ == "__main__":
    app.run(main)
