import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks import *

arch_part1 = [
    # 开头
    to_head('..'),
    to_cor(),
    to_begin(),

    # 添加输入层
    to_input('../examples/fcn8s/cats.jpg'),

    # MDDF Backbone
    # SimpleStem Block (128 channels, 3 channels in input)
    to_ConvConvRelu(name='SimpleStem', s_filer=128, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(2, 2),
                    height=40, depth=40),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(SimpleStem-east)", width=1, height=32, depth=32, opacity=0.5),

    # VSSBlock for 128 channels
    *block_2ConvPool(name='VSSBlock_128', botton='pool_b1', top='pool_b2', s_filer=256, n_filer=128, offset="(1,0,0)",
                     size=(32, 32, 3.5), opacity=0.5),

    # VisionClueMerge for 256 channels
    to_ConvConvRelu(name='VisionClueMerge_256', s_filer=32, n_filer=(512, 512), offset="(2,0,0)", to="(pool_b2-east)",
                    width=(8, 8), height=8, depth=8, caption="VisionClueMerge"),
    to_connection("pool_b2", "VisionClueMerge_256"),

    # VSSBlock for 256 channels
    *block_2ConvPool(name='VSSBlock_256', botton='VisionClueMerge_256', top='pool_b3', s_filer=128, n_filer=256,
                     offset="(1,0,0)", size=(25, 25, 4.5), opacity=0.5),

    # VisionClueMerge for 512 channels
    to_ConvConvRelu(name='VisionClueMerge_512', s_filer=64, n_filer=(1024, 1024), offset="(2,0,0)", to="(pool_b3-east)",
                    width=(8, 8), height=8, depth=8, caption="VisionClueMerge"),
    to_connection("pool_b3", "VisionClueMerge_512"),

    # VSSBlock for 512 channels
    *block_2ConvPool(name='VSSBlock_512', botton='VisionClueMerge_512', top='pool_b4', s_filer=256, n_filer=512,
                     offset="(1,0,0)", size=(16, 16, 5.5), opacity=0.5),

    # VisionClueMerge for 1024 channels
    to_ConvConvRelu(name='VisionClueMerge_1024', s_filer=64, n_filer=(2048, 2048), offset="(2,0,0)",
                    to="(pool_b4-east)", width=(8, 8), height=8, depth=8, caption="VisionClueMerge"),
    to_connection("pool_b4", "VisionClueMerge_1024"),

    # VSSBlock for 1024 channels
    *block_2ConvPool(name='VSSBlock_1024', botton='VisionClueMerge_1024', top='pool_b5', s_filer=512, n_filer=1024,
                     offset="(1,0,0)", size=(16, 16, 5.5), opacity=0.5),

    # DDF Block (512x512)
    to_ConvConvRelu(name='DDF_512', s_filer=128, n_filer=(512, 512), offset="(2,0,0)", to="(pool_b5-east)",
                    width=(8, 8), height=8, depth=8, caption="DDF"),
    to_connection("pool_b5", "DDF_512"),
]

arch_part2 = [
    # MDDF PAFPN Head
    # Upsample to P4 (nearest neighbor)
    to_ConvConvRelu(name="Upsample_P4", s_filer=256, n_filer=(128, 128), offset="(1,0,0)", to="(DDF_512-east)",
                    width=(2, 2), height=40, depth=40, caption="Upsample"),
    *block_2ConvPool(name='Concat_P4', botton='Upsample_P4', top='pool_concat', s_filer=128, n_filer=512,
                     offset="(1,0,0)", size=(16, 16, 5.5), opacity=0.5),

    # Upsample to P3
    to_ConvConvRelu(name="Upsample_P3", s_filer=256, n_filer=(128, 128), offset="(2,0,0)", to="(pool_concat-east)",
                    width=(2, 2), height=40, depth=40, caption="Upsample"),
    *block_2ConvPool(name='Concat_P3', botton='Upsample_P3', top='pool_concat3', s_filer=128, n_filer=256,
                     offset="(1,0,0)", size=(16, 16, 5.5), opacity=0.5),

    # Conv and Upsample to P4
    to_ConvConvRelu(name="Conv_256", s_filer=256, n_filer=(128, 128), offset="(1,0,0)", to="(pool_concat3-east)",
                    width=(3, 3), height=40, depth=40, caption="Conv"),
    to_ConvConvRelu(name="Final_Conv_256", s_filer=256, n_filer=(128, 128), offset="(2,0,0)", to="(pool_concat3-east)",
                    width=(3, 3), height=40, depth=40, caption="Final Conv"),

    # Final Detection Layer (softmax)
    to_ConvSoftMax(name="Softmax_Detection", s_filer=512, offset="(0.75,0,0)", to="(Final_Conv_256-east)", width=1,
                   height=40, depth=40, caption="Softmax Detection"),

    # 结束
    to_end()
]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    # Combine part1 and part2
    arch = arch_part1 + arch_part2
    to_generate(arch, namefile + '.tex')

if __name__ == '__main__':
    main()

