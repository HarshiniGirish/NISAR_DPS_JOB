cwlVersion: v1.2
class: CommandLineTool

label: nisar_access_subset
baseCommand: [python, /opt/app/app/nisar_access_subset.py]

inputs:
  access_mode:
    type: string
    inputBinding: { prefix: --access_mode }
    default: s3

  var:
    type: string
    inputBinding: { prefix: --var }
    default: HHHH

  row0:
    type: int
    inputBinding: { prefix: --row0 }
    default: 0
  row1:
    type: int
    inputBinding: { prefix: --row1 }
    default: 1024
  col0:
    type: int
    inputBinding: { prefix: --col0 }
    default: 0
  col1:
    type: int
    inputBinding: { prefix: --col1 }
    default: 1024

  out_dir:
    type: string
    inputBinding: { prefix: --out_dir }
    default: /tmp/output

outputs:
  subset_npy:
    type: File
    outputBinding:
      glob: "/tmp/output/*.npy"

requirements:
  InlineJavascriptRequirement: {}
