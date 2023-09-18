from omegaconf import OmegaConf
import hydra
from config import MainConfig, register_configs


register_configs()
@hydra.main(version_base=None, config_name="config")
def example_main(cfg: MainConfig) -> None:
    """I would have more specific name here. For example this file might be called create_arctic_dc.py and the main()
    function would be called create_arctic_db() or sth"""
    print(OmegaConf.to_yaml(cfg))
    # cfg_obj = OmegaConf.to_object(cfg)
    # print(cfg_obj.universe_.etfs)
    # print(cfg_obj.universe.equities)

    # cfg = MainConfig()
    # print(cfg.universe.equities)

if __name__ == "__main__":
    example_main()