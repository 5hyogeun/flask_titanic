from titanic.controller import TitanicController
from titanic.view import TitanicView
if __name__ == '__main__':
    def print_menu():
        print('0. EXIT')
        print('1. LEARNING MACHINE')
        print('2. VIEW : plot_survived_dead')
        print('2. TEST ACCURACY')
        return input('CHOOSE ONE \n')
    while 1:
        menu = print_menu()
        print('MENU : %s' % menu)
        if menu == '0':
            print('EXIT')
            break
        elif menu == '1':
            print('** CREATE TRAIN **')
            ctrl = TitanicController()
            t = ctrl.create_train()
            print('** t 모델 **')
            print(t)
            break
        elif menu == '2':
            view = TitanicView()
            t = view.create_train()
            # view.plot_survived_dead(t)
            # view.plot_sex(t)
            view.bar_chart(t, 'Pclass')
