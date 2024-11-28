from Models.MLP.Layer import Layer
from Models.MLP.Network import Mlp


def test_layer_feed_with_size():
    ob = Layer(False)
    ob.init_with_size(5)
    print(ob)


def test_layer_feed_with_values():
    ob = Layer(True)
    ob.init_with_values([.90, 0.1, 0.5])
    print(ob)


def test_neuron_init_weights():
    ob = Layer(True, next_layer_size=3)
    ob.init_with_size(3)
    for i in ob.neurons:
        print(i)


def test_network_add_input_output_layer():
    mlp = Mlp(learning_rate=0.5, epochs=200)
    mlp.lec_example1()
    mlp.fit(
        [
            [0, 0], [0, 1], [1, 0], [1, 1]
        ],
        [
            [0], [1], [1], [0]
        ]
    )


def do_work():
    test_network_add_input_output_layer()
