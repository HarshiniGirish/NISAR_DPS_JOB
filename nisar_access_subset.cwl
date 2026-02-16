cwlVersion: v1.2
class: CommandLineTool

label: nisar_access_subset

# Make sure your Dockerfile copies repo to /app (WORKDIR /app; COPY . /app)
baseCommand: [python, /app/nisar_access_subset.py]

inputs:
  access_mode:
    type: string
    inputBinding: { prefix: --access_mode }
    default: https

  https_href:
    type: string?
    inputBinding: { prefix: --https_href }

  s3_href:
    type: string?
    inputBinding: { prefix: --s3_href }

  vars:
    type: string
    inputBinding: { prefix: --vars }
    default: HHHH

  group:
    type: string
    inputBinding: { prefix: --group }
    default: /science/LSAR/GCOV/grids/frequencyA

  bbox:
    type: string?
    inputBinding: { prefix: --bbox }

  bbox_crs:
    type: string?
    inputBinding: { prefix: --bbox_crs }

  out_dir:
    type: string
    inputBinding: { prefix: --out_dir }
    default: /tmp/output

  out_name:
    type: string
    inputBinding: { prefix: --out_name }
    default: nisar_subset.zarr

outputs:
  zarr_store:
    type: Directory
    outputBinding:
      glob: $(inputs.out_dir + "/" + inputs.out_name)

  manifest:
    type: File
    outputBinding:
      glob: $(inputs.out_dir + "/manifest.json")

requirements:
  InlineJavascriptRequirement: {}
