FROM amazoncorretto:8

RUN yum update -y \
  && yum install -y \
  git \
  less \
  vim \
  maven \
  wget \
  tar \
  && yum clean all
  
# set up baseline user
ARG UID=1000
ARG GID=1000
RUN groupadd -o -g $GID baseline
RUN useradd -m -u $UID -g $GID -s /bin/bash baseline

# install CRAFT resources
WORKDIR /home/baseline
RUN wget https://github.com/UCDenver-ccp/CRAFT/archive/v4.0.1.tar.gz && \
    tar -xzvf v4.0.1.tar.gz

# copy code from this repository into the container
COPY src/ /home/baseline/code/src/
COPY pom.xml /home/baseline/code/
COPY data/ /home/baseline/data/

# transfer ownership to the baseline user
RUN chown -R baseline:baseline /home/baseline

USER baseline

# package/install code
RUN mvn clean install -f /home/baseline/code/pom.xml
RUN mkdir /home/baseline/dictionaries

ENV MAVEN_OPTS="-Xmx6G"

CMD ["/bin/sh", "-c", "mvn -f /home/baseline/code/pom.xml exec:java -Dexec.mainClass=edu.cuanschutz.ccp.mn_paper_baseline.BaselineFileGenerator -Dexec.args='/home/baseline/CRAFT-4.0.1 /home/baseline/dictionaries /home/baseline/data'"]