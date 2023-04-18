from jax import random

from ic_routing_board_generation.board_generator.numberlink_jax import Path as PathJax
from ic_routing_board_generation.board_generator.numberlink_jax import Mitm as MitmJax
from ic_routing_board_generation.board_generator.board_generator_numberlink_oj import Path as PathOld
from ic_routing_board_generation.board_generator.board_generator_numberlink_oj import Mitm as MitmOld

def test_Mitm():
    mitm = MitmOld(lr_price=2, t_price=1)
    # Using a larger path length in mitm might increase puzzle complexity, but
    # 8 or 10 appears to be the sweet spot if we want small sizes like 4x4 to
    # work.
    mitm.prepare(10)
    good_paths = [ PathOld(i for i in mitm._good_paths(0,0,0,1,20))]
    print("First 10 good paths are ", good_paths)
    for i in range(10):
        good_paths[i].show()
    return
    for _ in range(10):
        pathy = mitm.rand_path2(30, 30, 0, -1)
        print(type(pathy))
        print("Pathy is", pathy)
        print("showing pathy: ")
        pathy.show()

def test_MitmJax():
    # Generate a jax key
    key = random.PRNGKey(0)
    mitm = MitmJax(lr_price=2, t_price=1, key=key)
    # Using a larger path length in mitm might increase puzzle complexity, but
    # 8 or 10 appears to be the sweet spot if we want small sizes like 4x4 to
    # work.
    try:
        mitm.prepare(5)
    except Exception as e:
        print("Whoopsies")
    else:
        print(mitm)
        pathy = mitm.rand_path2(10, 10, 0, -1)
        print(type(mitm.rand_path2(10, 10, 0, -1)))

def test_MitmJaxErrors():
    # Generate a jax key
    key = random.PRNGKey(0)
    mitm = MitmJax(lr_price=2, t_price=1, key=key)
    # Using a larger path length in mitm might increase puzzle complexity, but
    # 8 or 10 appears to be the sweet spot if we want small sizes like 4x4 to
    # work.
    mitm.prepare(5)
    print("Whoopsies")
    print(mitm)
    pathy = mitm.rand_path2(10, 10, 0, -1)
    print(type(mitm.rand_path2(10, 10, 0, -1)))


if __name__ == '__main__':
    #test_Mitm()
    test_MitmJax()

    test_MitmJaxErrors()