"""Explicit model identities; there is intentionally no default final model."""
from dataclasses import dataclass
import re
from .joint_model import JointPARE, VARIANTS
from .controls import build_path_control, CONTROLS


@dataclass(frozen=True)
class ModelSpec:
    family: str
    variant: str | None = None
    abc: str | None = None
    control: str | None = None

    @classmethod
    def parse(cls, text):
        pieces=text.split(':')
        if len(pieces)==3 and pieces[0]=='path' and re.fullmatch('[01]{3}',pieces[1]) and pieces[2] in CONTROLS:
            return cls('path',abc=pieces[1],control=pieces[2])
        if len(pieces)==2 and pieces[0]=='joint' and pieces[1] in VARIANTS+['joint_stats_off']:
            return cls('joint',variant=pieces[1])
        raise ValueError('Explicit model required: path:ABC:control or joint:variant')

    def text(self):
        return f'path:{self.abc}:{self.control}' if self.family=='path' else f'joint:{self.variant}'

    def build(self, dimension, coarse_mean, fine_mean):
        if ModelSpec.parse(self.text())!=self:
            raise ValueError('Noncanonical or incomplete model specification')
        if self.family=='path':
            return build_path_control(dimension,self.abc,self.control,coarse_mean,fine_mean)
        if self.variant=='joint_stats_off':
            return build_path_control(dimension,'111','stats_off',coarse_mean,fine_mean)
        return JointPARE(dimension,self.variant,coarse_mean,fine_mean)

    def validate_metadata(self, metadata, model):
        variant=metadata.get('variant')
        if self.family=='path':
            accepted={f'path_{self.abc}__{self.control}'}
            if self.control=='full':accepted.add('path_'+self.abc)
            if variant not in accepted:raise ValueError('Checkpoint model does not match explicit ABC/control')
            if any(metadata.get(key,self.abc)!=self.abc for key in ['combination','abc','path_combination','requested_abc']) or any(metadata.get(key,self.control)!=self.control for key in ['control','requested_control']):
                raise ValueError('Conflicting checkpoint ABC/control fields')
            if 'paths_enabled' in metadata and metadata['paths_enabled']!=dict(zip('ABC',[x=='1' for x in self.abc])):
                raise ValueError('Conflicting checkpoint path flags')
        elif variant!=self.variant:
            raise ValueError('Checkpoint joint variant differs from explicit model')
        if 'ablate' in metadata and set(metadata['ablate'])!=model.physics.ablate:
            raise ValueError('Checkpoint component flags differ')
        if 'underlying_joint_variant' in metadata and metadata['underlying_joint_variant']!=model.variant:
            raise ValueError('Checkpoint underlying model differs')
