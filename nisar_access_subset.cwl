cwlVersion: v1.2
class: CommandLineTool
label: nisar_access_subset

baseCommand:
  - python
  - /opt/app/nisar_access_subset.py

inputs:
  access_mode:
    type: string
    default: auto
    inputBinding:
      prefix: --access_mode
  https_href:
    type: string?
    inputBinding:
      prefix: --https_href
  s3_href:
    type: string?
    inputBinding:
      prefix: --s3_href
  vars:
    type: string
    default: HHHH
    inputBinding:
      prefix: --vars
  group:
    type: string
    default: /science/LSAR/GCOV/grids/frequencyA
    inputBinding:
      prefix: --group
  bbox:
    type: string?
    inputBinding:
      prefix: --bbox
  bbox_crs:
    type: string?
    inputBinding:
      prefix: --bbox_crs
  out_name:
    type: string
    default: nisar_subset.zarr
    inputBinding:
      prefix: --out_name

outputs:
  out:
    type: Directory
    outputBinding:
      glob: output
