FROM jupyterhub/jupyterhub:latest

# Instala spawners o autenticadores si quieres
# RUN pip install dockerspawner

COPY jupyterhub_config.py /srv/jupyterhub/jupyterhub_config.py

CMD ["jupyterhub", "-f", "/srv/jupyterhub/jupyterhub_config.py"]

