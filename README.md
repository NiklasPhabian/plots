# Install


    mkvirtualenv --python=/usr/bin/python3 $PROJECT_ENV

    pip3 install matplotlib    
    pip3 install pandas
    pip3 install sqlalchemy

    git clone git@github.com:NiklasPhabian/plots.git $PLOTS_DIR
    pip3 install -e $PLOTS_DIR


# Usage
import plots

    plt = plots.Plot()
    a = [1,3,17]
    plt.plot(a)
    plt.show()
