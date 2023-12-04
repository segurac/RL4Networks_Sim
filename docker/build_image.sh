docker build -t ns3.30-mobility:latest ./


containerId=$(docker create ns3.30-mobility:latest)
docker cp "$containerId":/usr/ns-allinone-3.30/ns-3.30-mobility/contrib/opengym/model/ns3gym/dist/ns3gym-0.1.0-py3-none-any.whl ./
docker rm "$containerId"


