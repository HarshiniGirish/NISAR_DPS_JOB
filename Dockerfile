FROM mas.maap-project.org/root/maap-workspaces/custom_images/maap_base:v5.0.0

COPY . /opt/app

RUN chmod +x /opt/app/run.sh /opt/app/build.sh

ENTRYPOINT ["/opt/app/run.sh"]
